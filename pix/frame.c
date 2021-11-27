#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "pix.h"

/*-------------------------------------------------------------------------
*
*  Create or delete a frame data structure.
*
*------------------------------------------------------------------------*/

frameloc_t *frameCreate(para_t *p, int idx, matmx_t *mx, char type) {
    frameloc_t *fm;
    int sdim[4], elem_size=0;

    if ((fm = malloc(sizeof(frameloc_t))) == NULL)
        pstop("!!! frameCreate: not enough memory.\n");
    fm->ID    = idx;
    fm->type  = type;
    fm->dim_x = p->frame_x2 - p->frame_x1 + 1;
    fm->dim_y = p->frame_y2 - p->frame_y1 + 1;

    if (fm->dim_x > mx->dim_x)
        pstop("!!! frameCreate: image selected dim_x exceeds, max=%d\n",
              mx->dim_x);
    if (fm->dim_y > mx->dim_y)
        pstop("!!! frameCreate: image selected dim_x exceeds, max=%d\n",
              mx->dim_y);

    switch (type) {
    case 's':
        elem_size = sizeof(short);
        break;
    case 'i':
        elem_size = sizeof(int);
        break;
    case 'f':
        elem_size = sizeof(float);
        break;
    case 'd':
        elem_size = sizeof(double);
        break;
    default:
        pstop("!!! frameCreate: unkown frame type specified: %c\n", type);
    }

    if (!(fm->frame = malloc(fm->dim_x * fm->dim_y * elem_size)))
        pstop("!!! frameCreate: cannot create a new frameloc_t data.\n");
    if (mx != NULL) {
        sdim[0] = p->frame_x1;
        sdim[1] = p->frame_x2;
        sdim[2] = p->frame_y1;
        sdim[3] = p->frame_y2;
        mx_sub_s(mx->dim_x, mx->dim_y, sdim, (short*)(mx->data),
                 (short*)fm->frame);
    } else
        memset(fm->frame, 0, fm->dim_x*fm->dim_y*elem_size);

    return fm;
}

void frameDelete(frameloc_t *fm) {
    if (fm->frame) free(fm->frame);
    free(fm);
}

/*-------------------------------------------------------------------------
*
*  Compute the Spots initial coordinates.
*
*------------------------------------------------------------------------*/

static void push_spot(para_t *p, int mode, int fID, int x, int y, short *simg) {
    int m_sp, n_sp, nfsep, i, j, ii, dx, dy, df, imglen;
    sp_t **sp;

    if (mode == 0) {
        m_sp = p->m_sp1;
        n_sp = p->n_sp1;
        sp   = p->sp1;
    } else {
        m_sp = p->m_sp2;
        n_sp = p->n_sp2;
        sp   = p->sp2;
    }

// Allocate the index space.
    if (n_sp >= m_sp) {
        m_sp += 1024;
        if ((sp = realloc(sp, sizeof(sp_t *)*m_sp)) == NULL)
            pstop("!!! push_spot: not enough memory for sp.\n");
    }

// Search for the same spot within nfsep frames
    ii    = -1;
    nfsep = p->nfsep;
    for (i=n_sp-1; i >= 0; i--) {
        df = (sp[i] != NULL) ? sp[i]->fID - fID : 0;
        if (df <= 0) continue;
        if (df > nfsep) break;

        dx = abs(x - sp[i]->x);
        dy = abs(y - sp[i]->y);
        if (dx + dy <= 1) {
            ii = i;
            break;
        }
    }

    i = n_sp;
    n_sp++;
    imglen = p->x_find_pixels * p->y_find_pixels;
    if (ii < 0) {
// Initialize a new spot.
        if ((sp[i] = malloc(sizeof(sp_t)+imglen*sizeof(int))) == NULL)
            pstop("!!! push_spot: not enough memory for sp: %d\n", i);
        memset(sp[i], 0, sizeof(sp_t)+imglen*sizeof(int));
        sp[i]->x   = x;
        sp[i]->y   = y;
        sp[i]->fID = fID;
        sp[i]->img = (int *)(sp[i] + 1);
    } else {
// Move the found spot to the new frame position.
        sp[i]  = sp[ii];
        sp[ii] = NULL;
        sp[i]->fID = fID;
    }

// Summing the spot pixels for this frame.
    sp[i]->cnt++;
//#pragma omp parallel for private(j)
    
    for (j=0; j < imglen; j++)
        sp[i]->img[j] += (int)(simg[j]);

    if (mode == 0) {
        p->m_sp1 = m_sp;
        p->n_sp1 = n_sp;
        p->sp1   = sp;
    } else {
        p->m_sp2 = m_sp;
        p->n_sp2 = n_sp;
        p->sp2   = sp;
    }
}

/*-------------------------------------------------------------------------
*
*  Check whether spot contain many bright pixels
*
*  return:  -1:  invalid spot.
*            0:  normal spot.
*            1:  high intensity spot.
*
*------------------------------------------------------------------------*/

static int SpotCheck(para_t *p, short *intensity) {
    short *sBW;
    int cpos, sdim_x, sdim_y, imglen, i, k, imax, vmax;

    sdim_x = p->x_find_pixels;
    sdim_y = p->y_find_pixels;
    imglen = sdim_x * sdim_y;
    cpos   = (imglen-1)/2;      // center position of a spot window
    vmax   = vmax_s(imglen, intensity, &imax);
    if (imax != cpos) return -1;

    if ((sBW  = malloc(imglen*sizeof(short))) == NULL)
        pstop("!!! SpotCheck: not enough memory.\n");
    regional_max(sdim_x, sdim_y, 8, 's', intensity, sBW);
    for (i=0, k=0; i < imglen; i++) {
        if (intensity[i] < 0) intensity[i] = 0;
        if ((double)(intensity[i]) > p->threshold1 && sBW[i]==1) k++;
    }
    free(sBW);

    if (k > 1) return -1;

    return (vmax <= p->threshold2) ? 0 : 1;
}

/*-------------------------------------------------------------------------
*
*  Find the candidate spots: Using the regional maximum algorithm
*
*------------------------------------------------------------------------*/

void frameSpots(para_t *p, frameloc_t *fm0, frameloc_t *fm1) {
    int sdim[4], sdim_x, sdim_y, rng_x, rng_y;
    int dim_x, dim_y, x, y, xx, r;
    short  *frame0, *frame1, *dframe, *spot, *iBW;

    frame0 = fm0->frame;
    frame1 = (fm1) ? fm1->frame : NULL;
    dim_x  = fm0->dim_x;
    dim_y  = fm0->dim_y;
    sdim_x = p->x_find_pixels;
    sdim_y = p->y_find_pixels;
    rng_x  = sdim_x / 2;
    rng_y  = sdim_y / 2;
    iBW    = malloc(dim_x*dim_y*sizeof(short));
    spot   = malloc(sdim_x*sdim_y*sizeof(short));
    if (!iBW || !spot)
        pstop("!!! frameSpots: not enough memory.\n");

    if (frame1) {
// Obtain the differential image from (*this* - *next*) frames.
        if ((dframe = malloc(dim_x*dim_y*sizeof(short))) == NULL)
            pstop("!!! frameSpots: not enough memory for dframe.\n");
        vadd_s(dim_x*dim_y, 0, frame0, frame1, dframe);
    } else
        dframe = frame0;
    regional_max(dim_x, dim_y, 4, 's', dframe, iBW);

// Scan the frame and get the position for the candidate spots.
    for (y=0; y < dim_y; y++) {
        for (x=0; x < dim_x; x++) {
            xx = x + y*dim_x;
            if ((double)(dframe[xx]) <= p->threshold1 || iBW[xx] == 0) continue;

            sdim[0] = x-rng_x;
            sdim[1] = x+rng_x;
            sdim[2] = y-rng_y;
            sdim[3] = y+rng_y;
            mx_sub_s(dim_x, dim_y, sdim, dframe, spot);
            if ((r = SpotCheck(p, spot)) < 0) continue;

            if (frame1)
                mx_rsub_s(dim_x, dim_y, sdim, frame1, frame0);
            if (r == 0)
                push_spot(p, 0, fm0->ID, x, y, spot);
            else if (p->outfnH != NULL)
                push_spot(p, 1, fm0->ID, x, y, spot);
        }
    }

    free(spot);
    free(iBW);
    if (frame1) free(dframe);
}

/*-------------------------------------------------------------------------
*
*  Find the candidate spots: Using the max intensity and mark algorithm
*
*------------------------------------------------------------------------*/

typedef struct {
    short II;
    int x, y;
} sp_sort_t;

static int pixel_cmp(const void *a, const void *b) {
    sp_sort_t *aa = (sp_sort_t *)a;
    sp_sort_t *bb = (sp_sort_t *)b;

    if (aa->II > bb->II)
        return -1;
    else if (aa->II < bb->II)
        return 1;
    else
        return 0;
}

static int
spot_sort(int Imin, int dim_x, int dim_y, short *dframe, sp_sort_t *sps) {
    int i, j, k, k0;
    short II;

    k = 0;
    for (j=0; j < dim_y; j++) {
        for (i=0; i < dim_x; i++) {
            k0 = i+j*dim_x;
            II = dframe[k0];
            if (II > Imin) {
                sps[k].x  = i;
                sps[k].y  = j;
                sps[k].II = II;
                k++;
            }
        }
    }
    qsort(sps, k, sizeof(sp_sort_t), pixel_cmp);

    return k;
}

static int check_mark(int x0, int y0, int wx, int wy, int dim_x, int dim_y,
                      short *mark) {
    int xx, x, y;

    if (x0-wx < 0 || y0-wy < 0 || x0+wx >= dim_x || y0+wy >= dim_y) return -1;

    for (y=y0-wy; y <= y0+wy; y++) {
        for (x=x0-wx; x <= x0+wx; x++) {
            xx = x + y*dim_x;
            if (mark[xx] != 0) return -1;
        }
    }

//#pragma omp parallel for private(x,y,xx)
    for (y=y0-wy; y <= y0+wy; y++) {
        for (x=x0-wx; x <= x0+wx; x++) {
            xx = x + y*dim_x;
            mark[xx] = (short)1;
        }
    }
    return 0;
}

void frameSpot2(para_t *p, frameloc_t *fm0, frameloc_t *fm1) {
    int sdim[4], sdim_x, sdim_y, rng_x, rng_y;
    int dim_x, dim_y, nsp, x, y, i;
    short     *frame0, *frame1, *dframe, *spot, *mark, II;
    sp_sort_t *sps;

    frame0 = fm0->frame;
    frame1 = (fm1) ? fm1->frame : NULL;
    dim_x  = fm0->dim_x;
    dim_y  = fm0->dim_y;
    sdim_x = p->x_find_pixels;
    sdim_y = p->y_find_pixels;
    rng_x  = sdim_x / 2;
    rng_y  = sdim_y / 2;
    sps    = malloc(dim_x*dim_y*sizeof(sp_sort_t));
    spot   = malloc(sdim_x*sdim_y*sizeof(short));
    mark   = calloc(dim_x*dim_y, sizeof(short));
    if (!sps || !mark || !spot)
        pstop("!!! frameSpot2: not enough memory.\n");

    if (frame1) {
// Obtain the differential image from (*this* - *next*) frames.
        if ((dframe = malloc(dim_x*dim_y*sizeof(short))) == NULL)
            pstop("!!! frameSpots: not enough memory for dframe.\n");
        vadd_s(dim_x*dim_y, 0, frame0, frame1, dframe);
    } else
        dframe = frame0;
    nsp = spot_sort(p->threshold1, dim_x, dim_y, dframe, sps);

// Scan the frame and get the position for the candidate spots.
    for (i=0; i < nsp; i++) {
        x  = sps[i].x;
        y  = sps[i].y;
        II = sps[i].II;
        if ((double)II <= p->threshold1) break;
        if (check_mark(x, y, rng_x, rng_y, dim_x, dim_y, mark) != 0) continue;

        sdim[0] = x-rng_x;
        sdim[1] = x+rng_x;
        sdim[2] = y-rng_y;
        sdim[3] = y+rng_y;
        mx_sub_s(dim_x, dim_y, sdim, dframe, spot);
        if (frame1)
            mx_rsub_s(dim_x, dim_y, sdim, frame1, frame0);
        if ((double)II <= p->threshold2)
            push_spot(p, 0, fm0->ID, x, y, spot);
        else if (p->outfnH != NULL)
            push_spot(p, 1, fm0->ID, x, y, spot);
    }
    free(spot);
    free(mark);
    free(sps);
    if (frame1) free(dframe);
}

/*-------------------------------------------------------------------------
*
*  Accumulate one frame.
*
*------------------------------------------------------------------------*/

void frame_sum(para_t *p, frameloc_t *fm) {
    int npx, i;
    short *f;

    f = (short *)(fm->frame);
    npx = (p->frame_x2 - p->frame_x1 + 1) * (p->frame_y2 - p->frame_y1 + 1);
    for (i=0; i < npx; i++)
        p->psum[i] += ((int)f[i]);
}
