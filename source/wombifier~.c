// wombifier~.c — Max/MSP external: "in utero" with optional watery texture
// Version: PrimeFIR-enabled (linear-phase FIR with prime-index sparsification)
// Author: You + ChatGPT
// License: MIT
// Build: drop in a Max SDK MSP external project; compile 64-bit

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"
#include "ext_sysmem.h"
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------- Biquad (RBJ) ---------------------------------
typedef struct _biquad {
    double b0, b1, b2, a1, a2;
    double x1, x2, y1, y2;
} t_biquad;

static void biquad_clear(t_biquad *b) { b->x1 = b->x2 = b->y1 = b->y2 = 0.0; }

static void biquad_set_lp(t_biquad *b, double sr, double cutoff, double q) {
    if (cutoff < 20.0) cutoff = 20.0;
    if (cutoff > sr * 0.45) cutoff = sr * 0.45;
    if (q < 0.1) q = 0.1;

    double w0 = 2.0 * M_PI * (cutoff / sr);
    double cosw0 = cos(w0);
    double sinw0 = sin(w0);
    double alpha = sinw0 / (2.0 * q);

    double b0 = (1.0 - cosw0) * 0.5;
    double b1 = 1.0 - cosw0;
    double b2 = (1.0 - cosw0) * 0.5;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    b->b0 = b0 / a0; b->b1 = b1 / a0; b->b2 = b2 / a0;
    b->a1 = a1 / a0; b->a2 = a2 / a0;
}

static inline double biquad_process(t_biquad *b, double x) {
    double y = b->b0 * x + b->b1 * b->x1 + b->b2 * b->x2 - b->a1 * b->y1 - b->a2 * b->y2;
    b->x2 = b->x1; b->x1 = x;
    b->y2 = b->y1; b->y1 = y;
    return y;
}

// ---- RBJ bandpass corretto (constant skirt gain, peak gain = Q) ------------
static void biquad_set_bp(t_biquad *b, double sr, double fc, double q) {
    if (fc < 20.0) fc = 20.0;
    if (fc > sr * 0.45) fc = sr * 0.45;
    if (q < 0.1) q = 0.1;

    double w0 = 2.0 * M_PI * (fc / sr);
    double cosw0 = cos(w0);
    double sinw0 = sin(w0);
    double alpha = sinw0 / (2.0 * q);

    double b0 = q * alpha;   // = sin(w0)/2
    double b1 = 0.0;
    double b2 = -q * alpha;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    b->b0 = b0 / a0; b->b1 = b1 / a0; b->b2 = b2 / a0;
    b->a1 = a1 / a0; b->a2 = a2 / a0;
    biquad_clear(b);
}

// ---- Lowshelf (per MTF/voice, opzionali) -----------------------------------
static void biquad_set_lowshelf(t_biquad *b, double sr, double fc, double db_gain) {
    double A = pow(10.0, db_gain/40.0);
    double w0 = 2.0 * M_PI * (fc/sr);
    double cosw0 = cos(w0);
    double sinw0 = sin(w0);
    double alpha = sinw0/2.0 * sqrt((A + 1.0/A) * (1.0/0.707 - 1.0) + 2.0);

    double b0 = A*((A+1.0) - (A-1.0)*cosw0 + 2.0*sqrt(A)*alpha);
    double b1 = 2.0*A*((A-1.0) - (A+1.0)*cosw0);
    double b2 = A*((A+1.0) - (A-1.0)*cosw0 - 2.0*sqrt(A)*alpha);
    double a0 = (A+1.0) + (A-1.0)*cosw0 + 2.0*sqrt(A)*alpha;
    double a1 = -2.0*((A-1.0) + (A+1.0)*cosw0);
    double a2 = (A+1.0) + (A-1.0)*cosw0 - 2.0*sqrt(A)*alpha;

    b->b0 = b0/a0; b->b1 = b1/a0; b->b2 = b2/a0;
    b->a1 = a1/a0; b->a2 = a2/a0;
}

// -------------------------- Resonator Bank ----------------------------------
typedef struct _resonator {
    t_biquad bands[3];
    double gains[3];
} t_resonator;

static void resonator_init(t_resonator *r, double sr) {
    double freqs[3] = {200.0, 400.0, 600.0};
    double qs[3]    = {4.0,   3.5,   3.0};
    double gains[3] = {1.0,   0.7,   0.5};
    for (int i = 0; i < 3; i++) {
        biquad_set_bp(&r->bands[i], sr, freqs[i], qs[i]);
        r->gains[i] = gains[i];
    }
}

static inline double resonator_process(t_resonator *r, double in) {
    double out = 0.0;
    for (int i = 0; i < 3; i++)
        out += biquad_process(&r->bands[i], in) * r->gains[i];
    return out;
}

// -------------------------- FIR PrimeFIR-like --------------------------------
// Sinc finestrato Blackman-Harris con possibilità di usare solo indici PRIMI
typedef struct _fir {
    double *taps;  // coeff
    int ntaps;     // dispari
    double *buf;   // ring buffer
    int idx;
    double gain_l1, gain_l2;
} t_fir;

static int is_prime_i(int n) {
    if (n < 2) return 0; if ((n & 1) == 0) return n == 2;
    for (int k = 3; k * k <= n; k += 2) if (n % k == 0) return 0;
    return 1;
}

static void fir_free(t_fir *f) {
    if (f->taps) sysmem_freeptr(f->taps);
    if (f->buf)  sysmem_freeptr(f->buf);
    memset(f, 0, sizeof(*f));
}

static void fir_prepare(t_fir *f, int ntaps) {
    int wanted = (ntaps % 2 == 0) ? (ntaps + 1) : ntaps;
    if (wanted < 3) wanted = 3;

    if (!f->taps || !f->buf || f->ntaps != wanted) {
        if (f->taps) sysmem_freeptr(f->taps);
        if (f->buf)  sysmem_freeptr(f->buf);
        f->taps = (double*)sysmem_newptrclear(sizeof(double) * wanted);
        f->buf  = (double*)sysmem_newptrclear(sizeof(double) * wanted);
        f->ntaps = wanted;
    } else {
        memset(f->buf, 0, sizeof(double) * f->ntaps);
    }

    f->idx = 0;
}

static void fir_make_lowpass(t_fir *f, double sr, double fc,
                             int prime_mode, double asym, int energy_norm) {
    int N = f->ntaps, M = N - 1;
    double fc_n = fc / (sr * 0.5); // [0..1]
    if (fc_n < 0.001) fc_n = 0.001;
    if (fc_n > 0.999) fc_n = 0.999;

    // Centro asimmetrico: 0.5 = simmetrico; <0.5 => più post-ring
    double center = (double)M * (0.5 - 0.25 * asym);

    const double a0 = 0.35875, a1 = 0.48829, a2 = 0.14128, a3 = 0.01168; // BH4

    double sumL1 = 0.0, sumL2 = 0.0;
    int cidx = (int)floor(center + 0.5);

    for (int n = 0; n < N; ++n) {
        double k = (double)n - center;
        double x = M_PI * k * fc_n;
        double sinc = (fabs(x) < 1e-12) ? fc_n : sin(x) / (M_PI * k);

        double warg = (double)n / (double)M;
        double w = a0 - a1*cos(2.0*M_PI*warg) + a2*cos(4.0*M_PI*warg) - a3*cos(6.0*M_PI*warg);

        double tap = sinc * w;

        if (prime_mode && n != cidx) {
            int mirror = M - n;
            if (!(is_prime_i(n) || is_prime_i(mirror))) {
                tap = 0.0;
            }
        }

        f->taps[n] = tap;
        sumL1 += tap;
        sumL2 += tap * tap;
    }

    if (prime_mode) {
        for (int n = 0; n < N/2; ++n) {
            int m = M - n;
            double avg = 0.5 * (f->taps[n] + f->taps[m]);
            f->taps[n] = f->taps[m] = avg;
        }
        sumL1 = 0.0;
        sumL2 = 0.0;
        for (int n = 0; n < N; ++n) {
            double tap = f->taps[n];
            sumL1 += tap;
            sumL2 += tap * tap;
        }
    }

    // Normalizzazione
    if (energy_norm) {
        double norm = sqrt(sumL2);
        if (norm > 1e-15) for (int n=0; n<N; ++n) f->taps[n] /= norm;
        f->gain_l1 = 0.0; f->gain_l2 = 1.0;
    } else {
        if (fabs(sumL1) < 1e-15) sumL1 = 1.0;
        for (int n=0; n<N; ++n) f->taps[n] /= sumL1; // DC = 1.0
        f->gain_l1 = 1.0; f->gain_l2 = 0.0;
    }

    memset(f->buf, 0, sizeof(double)*N);
    f->idx = 0;
}

static inline double fir_process_1(t_fir *f, double x) {
    int N = f->ntaps;
    f->buf[f->idx] = x;
    double y = 0.0;
    int i = f->idx;
    for (int n = 0; n < N; ++n) {
        if (--i < 0) i += N;
        y += f->taps[n] * f->buf[i];
    }
    if (++f->idx >= N) f->idx = 0;
    return y;
}

// ------------------------------ Helpers -------------------------------------
static inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline double softclip(double x, double amt) { // amt 0..1
    if (amt <= 0.0) return x;
    double drive = 1.0 + amt * 12.0;
    double y = tanh(drive * x);
    double norm = tanh(drive);
    return (norm > 0.0) ? (y / norm) : y;
}

static inline double rand_uniform_u32(unsigned int *state) { // [0,1)
    *state ^= *state << 13; *state ^= *state >> 17; *state ^= *state << 5;
    return (*state) * (1.0 / 4294967296.0);
}

static inline double rand_uniform_pm1(unsigned int *state) { // [-1,1]
    return 2.0 * rand_uniform_u32(state) - 1.0;
}

// Inviluppo cardiaco a area costante (media ~1 per ciclo)
static double heart_env_area_norm(double mu1, double s1, double mu2, double s2, double g2, double phase) {
    // gaussiane su [0,1) senza wrap elaborata (sufficiente per sigma piccole)
    double g1 = exp(-0.5 * pow((phase - mu1) / s1, 2.0));
    double g2v = exp(-0.5 * pow((phase - mu2) / s2, 2.0)) * g2;
    double e  = g1 + g2v;
    // area ≈ sqrt(2π) * (s1 + s2*g2)
    double area = 2.5066282746310002 * (s1 + s2 * g2);
    if (area > 1e-9) e /= area; // media ≈ 1 su un ciclo
    // soft-knee leggero che preserva mediamente l'area
    e = e / (1.0 + 0.5 * e);
    if (e > 3.0) e = 3.0;
    return e;
}

// ------------------------------ Object --------------------------------------
typedef struct _wombifier {
    t_pxobject x_obj;

    // base params
    double bpm;        // 20..200 (default 72)
    double depth;      // 0..1   (AM depth)
    double cutoff;     // Hz
    double q;          // Q
    double noise_amt;  // 0..1
    double wet;        // 0..1

    // extras
    double cutmod;     // 0..1 cutoff modulation by heartbeat (IIR path)
    double soft;       // 0..1 soft saturator amount

    // watery/chorus
    double water;      // 0..1
    double water_ms;   // 1..30
    double water_rate; // Hz
    double stereo_ms;  // 0..20 (scaled by width)

    // new parameters
    double thickness;  // 0..1
    double warmth;     // 0..1
    double bodyres;    // 0..1
    double feedback;   // 0..0.3
    double dynamics;   // 0..1
    double width;      // 0..1 stereo width (0 = perfetta simmetria)
    long   mono;       // forza input mono

    // dsp state
    double sr;
    double phase;      // heartbeat phase
    double gauss1_mu, gauss1_sigma;
    double gauss2_mu, gauss2_sigma, gauss2_gain;

    // noise
    unsigned int rng;
    double noise_lp;   // one-pole state
    double noise_g;    // one-pole g
    t_biquad noise_filter[2];

    // LPF IIR occlusion (solo se FIR off)
    t_biquad lpfL, lpfR;
    double cutoff_smooth; // runtime smoothed cutoff
    char coeffs_dirty;

    // watery delay
    double *dlyL, *dlyR;
    long dly_size;
    long wr_idx;
    double lfoL, lfoR;

    char  in_connected[2]; // signal inlet connection flags

    // resonators
    t_resonator resL, resR;

    // envelope follower
    double env_follow;
    double env_smooth;

    // optional research-based filters (toggle)
    t_biquad mtf_filter[2];
    t_biquad voice_enhance[2];
    long mtf_on;
    long mvoice_on;

    // HRV / respiration
    double hrv_phase, resp_phase;
    double hrv_rate, resp_rate;

    // Level monitoring
    double leq_acc;
    double peak_level;
    long   leq_count;

    // FIR PrimeFIR-like
    long   use_fir;        // 0/1
    long   fir_ntaps;      // odd
    long   fir_prime;      // 0/1 only primes
    double fir_asym;       // 0..1
    long   fir_energy_norm;// 0=DC norm, 1=energy norm
    t_fir  firL, firR;
    char   fir_dirty;

} t_wombifier;

// class pointer
t_class *wombifier_class;

// prototypes
void *wombifier_new(t_symbol *s, long argc, t_atom *argv);
void wombifier_free(t_wombifier *x);
void wombifier_assist(t_wombifier *x, void *b, long m, long a, char *s);
void wombifier_dsp64(t_wombifier *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void wombifier_perform64(t_wombifier *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long n, long flags, void *userparam);
void wombifier_anything(t_wombifier *x, t_symbol *s, long argc, t_atom *argv);
void wombifier_set_fir_enabled(t_wombifier *x, void *attr, long ac, t_atom *av);

static void biquad_set_womb_response(t_biquad *b, double sr) {
    // opzionale: forte attenuazione sopra 1k
    biquad_set_lowshelf(b, sr, 1000.0, -24.0);
}
static void biquad_set_maternal_voice(t_biquad *b, double sr) {
    // opzionale: lieve boost 500 Hz
    biquad_set_lowshelf(b, sr, 500.0, 3.0);
}

static void ensure_delay(t_wombifier *x) {
    long need = (long)ceil((x->sr * 0.100) + 8.0);
    if (need < 256) need = 256;
    if (need != x->dly_size) {
        if (x->dlyL) sysmem_freeptr(x->dlyL);
        if (x->dlyR) sysmem_freeptr(x->dlyR);
        x->dlyL = (double *)sysmem_newptrclear(sizeof(double) * need);
        x->dlyR = (double *)sysmem_newptrclear(sizeof(double) * need);
        x->dly_size = need;
        x->wr_idx = 0;
    }
}

static inline double tap_delay(const double *buf, long size, long wr_idx, double delay_samps) {
    if (delay_samps < 1.0) delay_samps = 1.0;
    if (delay_samps > size - 4) delay_samps = size - 4;

    double read_pos = (double)wr_idx - delay_samps;
    while (read_pos < 0.0) read_pos += size;

    long i1 = (long)read_pos;
    long i0 = i1 - 1; if (i0 < 0) i0 += size;
    long i2 = i1 + 1; if (i2 >= size) i2 -= size;
    long i3 = i2 + 1; if (i3 >= size) i3 -= size;

    double frac = read_pos - (double)i1;
    double a0 = buf[i0], a1 = buf[i1], a2 = buf[i2], a3 = buf[i3];

    double c0 = a1;
    double c1 = 0.5 * (a2 - a0);
    double c2 = a0 - 2.5 * a1 + 2.0 * a2 - 0.5 * a3;
    double c3 = -0.5 * a0 + 1.5 * a1 - 1.5 * a2 + 0.5 * a3;

    return ((c3 * frac + c2) * frac + c1) * frac + c0;
}

static void update_coeffs(t_wombifier *x, double cutoff_override) {
    double c = cutoff_override > 0 ? cutoff_override : x->cutoff;
    biquad_set_lp(&x->lpfL, x->sr, c, x->q);
    biquad_set_lp(&x->lpfR, x->sr, c, x->q);
    x->coeffs_dirty = 0;
}

// Attribute setters
void wombifier_set_cutoff(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        double v = atom_getfloat(av);
        x->cutoff = clampd(v, 40.0, 4000.0);
        x->cutoff_smooth = x->cutoff;
        x->coeffs_dirty = 1;
        x->fir_dirty = 1; // FIR dipende dal cutoff
    }
}
void wombifier_set_q(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        double v = atom_getfloat(av);
        x->q = clampd(v, 0.3, 8.0);
        x->coeffs_dirty = 1;
    }
}
void wombifier_set_fir_enabled(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        long v = atom_getlong(av) ? 1 : 0;
        if (x->use_fir != v) {
            x->use_fir = v;
            x->fir_dirty = 1;
        } else {
            x->use_fir = v;
        }
    }
}
// FIR setters -> marcano fir_dirty
void wombifier_set_fir_ntaps(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        long v = atom_getlong(av);
        if (v < 129) v = 129; if (v > 4097) v = 4097;
        x->fir_ntaps = v | 1; // forza dispari
        x->fir_dirty = 1;
    }
}
void wombifier_set_fir_asym(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) { x->fir_asym = clampd(atom_getfloat(av), 0.0, 1.0); x->fir_dirty = 1; }
}
void wombifier_set_fir_prime(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) { x->fir_prime = atom_getlong(av) ? 1 : 0; x->fir_dirty = 1; }
}
void wombifier_set_fir_energy(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) { x->fir_energy_norm = atom_getlong(av) ? 1 : 0; x->fir_dirty = 1; }
}

// ---------------------------------- Main ------------------------------------
C74_EXPORT void ext_main(void *r) {
    t_class *c = class_new("wombifier~", (method)wombifier_new, (method)wombifier_free,
                           (long)sizeof(t_wombifier), 0L, A_GIMME, 0);

    post("wombifier~ (PrimeFIR) — build %s %s", __DATE__, __TIME__);

    class_dspinit(c);

    class_addmethod(c, (method)wombifier_assist, "assist", A_CANT, 0);
    class_addmethod(c, (method)wombifier_dsp64,  "dsp64",  A_CANT, 0);
    class_addmethod(c, (method)wombifier_anything, "anything", A_GIMME, 0);

    // base params
    CLASS_ATTR_DOUBLE(c, "bpm", 0, t_wombifier, bpm);
    CLASS_ATTR_LABEL (c, "bpm", 0, "Heart Rate (BPM)");
    CLASS_ATTR_FILTER_CLIP(c, "bpm", 20.0, 200.0);

    CLASS_ATTR_DOUBLE(c, "depth", 0, t_wombifier, depth);
    CLASS_ATTR_LABEL (c, "depth", 0, "Pulse Depth (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "depth", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "cutoff", 0, t_wombifier, cutoff);
    CLASS_ATTR_LABEL (c, "cutoff", 0, "Low-Pass Cutoff (Hz)");
    CLASS_ATTR_ACCESSORS(c, "cutoff", NULL, wombifier_set_cutoff);
    CLASS_ATTR_FILTER_CLIP(c, "cutoff", 40.0, 4000.0);

    CLASS_ATTR_DOUBLE(c, "q", 0, t_wombifier, q);
    CLASS_ATTR_LABEL (c, "q", 0, "Low-Pass Resonance (Q)");
    CLASS_ATTR_ACCESSORS(c, "q", NULL, wombifier_set_q);
    CLASS_ATTR_FILTER_CLIP(c, "q", 0.3, 8.0);

    CLASS_ATTR_DOUBLE(c, "noise", 0, t_wombifier, noise_amt);
    CLASS_ATTR_LABEL (c, "noise", 0, "Fluid Noise (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "noise", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "wet", 0, t_wombifier, wet);
    CLASS_ATTR_LABEL (c, "wet", 0, "Wet/Dry (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "wet", 0.0, 1.0);

    // extras
    CLASS_ATTR_DOUBLE(c, "cutmod", 0, t_wombifier, cutmod);
    CLASS_ATTR_LABEL (c, "cutmod", 0, "Cutoff Mod by Heartbeat (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "cutmod", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "soft", 0, t_wombifier, soft);
    CLASS_ATTR_LABEL (c, "soft", 0, "Soft Limiter Amount (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "soft", 0.0, 1.0);

    // watery / stereo
    CLASS_ATTR_DOUBLE(c, "water", 0, t_wombifier, water);
    CLASS_ATTR_LABEL (c, "water", 0, "Watery Mix (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "water", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "water_ms", 0, t_wombifier, water_ms);
    CLASS_ATTR_LABEL (c, "water_ms", 0, "Watery Base Delay (ms)");
    CLASS_ATTR_FILTER_CLIP(c, "water_ms", 1.0, 30.0);

    CLASS_ATTR_DOUBLE(c, "water_rate", 0, t_wombifier, water_rate);
    CLASS_ATTR_LABEL (c, "water_rate", 0, "Watery LFO Rate (Hz)");
    CLASS_ATTR_FILTER_CLIP(c, "water_rate", 0.01, 5.0);

    CLASS_ATTR_DOUBLE(c, "stereo_ms", 0, t_wombifier, stereo_ms);
    CLASS_ATTR_LABEL (c, "stereo_ms", 0, "Static Right Delay (ms) (scaled by width)");
    CLASS_ATTR_FILTER_CLIP(c, "stereo_ms", 0.0, 20.0);

    CLASS_ATTR_DOUBLE(c, "thickness", 0, t_wombifier, thickness);
    CLASS_ATTR_LABEL (c, "thickness", 0, "Body Thickness (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "thickness", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "warmth", 0, t_wombifier, warmth);
    CLASS_ATTR_LABEL (c, "warmth", 0, "Harmonic Warmth (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "warmth", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "bodyres", 0, t_wombifier, bodyres);
    CLASS_ATTR_LABEL (c, "bodyres", 0, "Body Resonance (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "bodyres", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "feedback", 0, t_wombifier, feedback);
    CLASS_ATTR_LABEL (c, "feedback", 0, "Water Feedback (0-0.3)");
    CLASS_ATTR_FILTER_CLIP(c, "feedback", 0.0, 0.3);

    CLASS_ATTR_DOUBLE(c, "dynamics", 0, t_wombifier, dynamics);
    CLASS_ATTR_LABEL (c, "dynamics", 0, "Dynamic Response (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "dynamics", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "width", 0, t_wombifier, width);
    CLASS_ATTR_LABEL (c, "width", 0, "Stereo Width (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "width", 0.0, 1.0);

    CLASS_ATTR_LONG (c, "mono", 0, t_wombifier, mono);
    CLASS_ATTR_LABEL(c, "mono", 0, "Force Mono Input (0/1)");

    // FIR attrs
    CLASS_ATTR_LONG (c, "fir", 0, t_wombifier, use_fir);
    CLASS_ATTR_LABEL(c, "fir", 0, "Enable FIR (0/1)");
    CLASS_ATTR_ACCESSORS(c, "fir", NULL, wombifier_set_fir_enabled);

    CLASS_ATTR_LONG (c, "fir_ntaps", 0, t_wombifier, fir_ntaps);
    CLASS_ATTR_LABEL(c, "fir_ntaps", 0, "FIR Taps (odd, 129..4097)");
    CLASS_ATTR_ACCESSORS(c, "fir_ntaps", NULL, wombifier_set_fir_ntaps);

    CLASS_ATTR_DOUBLE(c, "fir_asym", 0, t_wombifier, fir_asym);
    CLASS_ATTR_LABEL (c, "fir_asym", 0, "FIR Asymmetry (0..1)");
    CLASS_ATTR_ACCESSORS(c, "fir_asym", NULL, wombifier_set_fir_asym);

    CLASS_ATTR_LONG (c, "fir_prime", 0, t_wombifier, fir_prime);
    CLASS_ATTR_LABEL(c, "fir_prime", 0, "FIR Prime Mode (0/1)");
    CLASS_ATTR_ACCESSORS(c, "fir_prime", NULL, wombifier_set_fir_prime);

    CLASS_ATTR_LONG (c, "fir_energy", 0, t_wombifier, fir_energy_norm);
    CLASS_ATTR_LABEL(c, "fir_energy", 0, "Normalize Energy (0=DC,1=Energy)");
    CLASS_ATTR_ACCESSORS(c, "fir_energy", NULL, wombifier_set_fir_energy);

    // MTF / Voice toggles
    CLASS_ATTR_LONG (c, "mtf", 0, t_wombifier, mtf_on);
    CLASS_ATTR_LABEL(c, "mtf", 0, "Maternal Transfer (0/1)");

    CLASS_ATTR_LONG (c, "mvoice", 0, t_wombifier, mvoice_on);
    CLASS_ATTR_LABEL(c, "mvoice", 0, "Maternal Voice Enhance (0/1)");

    class_register(CLASS_BOX, c);
    wombifier_class = c;
}

// --------------------------------- New/Free ---------------------------------
void *wombifier_new(t_symbol *s, long argc, t_atom *argv) {
    t_wombifier *x = (t_wombifier *)object_alloc(wombifier_class);
    if (!x) return NULL;

    dsp_setup((t_pxobject *)x, 2);
    outlet_new((t_object *)x, "signal");
    outlet_new((t_object *)x, "signal");
    inlet_new((t_object *)x, NULL); // control inlet

    x->sr        = sys_getsr(); if (x->sr <= 0) x->sr = 48000.0;
    x->bpm       = 72.0;
    x->depth     = 0.6;
    x->cutoff    = 700.0;  // più realistico per occlusione
    x->q         = 1.2;
    x->noise_amt = 0.15;
    x->wet       = 1.0;

    x->cutmod    = 0.35;
    x->soft      = 0.2;

    x->water     = 0.35;
    x->water_ms  = 12.0;
    x->water_rate= 0.25;
    x->stereo_ms = 0.0; // niente offset fisso (si scala con width)

    x->thickness = 0.3;
    x->warmth    = 0.3;
    x->bodyres   = 0.25; // un po' meno, più naturale
    x->feedback  = 0.12;
    x->dynamics  = 0.25;
    x->width     = 0.0;  // perfetta simmetria di default
    x->mono      = 0;

    x->env_follow = 0.0;
    x->env_smooth = 0.99;

    x->phase      = 0.0;
    x->gauss1_mu = 0.05; x->gauss1_sigma = 0.04;
    x->gauss2_mu = 0.32; x->gauss2_sigma = 0.05; x->gauss2_gain = 0.7;

    x->rng = 222222227u;

    x->noise_lp = 0.0;
    x->noise_g = 1.0 - exp(-2.0 * M_PI * (300.0 / x->sr));
    biquad_set_lp(&x->noise_filter[0], x->sr, 600.0, 0.85);
    biquad_set_lp(&x->noise_filter[1], x->sr, 300.0, 0.75);

    biquad_clear(&x->lpfL);
    biquad_clear(&x->lpfR);
    update_coeffs(x, 0.0);
    x->cutoff_smooth = x->cutoff;
    x->coeffs_dirty = 0;

    resonator_init(&x->resL, x->sr);
    resonator_init(&x->resR, x->sr);

    // research-based filters prepared (OFF by default)
    biquad_set_womb_response(&x->mtf_filter[0], x->sr);
    biquad_set_womb_response(&x->mtf_filter[1], x->sr);
    biquad_set_maternal_voice(&x->voice_enhance[0], x->sr);
    biquad_set_maternal_voice(&x->voice_enhance[1], x->sr);
    x->mtf_on = 0;
    x->mvoice_on = 0;

    // modulation rates
    x->hrv_rate  = 0.1;
    x->resp_rate = 0.25;
    x->hrv_phase = 0.0;
    x->resp_phase= 0.0;

    // level monitoring
    x->leq_acc = 0.0; x->peak_level = 0.0; x->leq_count = 0;

    // delay
    x->dlyL = x->dlyR = NULL; x->dly_size = 0; x->wr_idx = 0;
    x->lfoL = 0.0; x->lfoR = 0.0; // stesse fasi per simmetria
    x->in_connected[0] = x->in_connected[1] = 0;
    ensure_delay(x);

    // FIR defaults — già pronti per "filtro numeri primi"
    x->use_fir        = 1;
    x->fir_ntaps      = 513;
    x->fir_prime      = 1;
    x->fir_asym       = 0.3;
    x->fir_energy_norm= 0; // DC normalization (= area costante)
    x->fir_dirty      = 1;
    x->firL.taps = x->firR.taps = NULL;
    x->firL.buf  = x->firR.buf  = NULL;

    attr_args_process(x, (short)argc, argv); // allow @attrs in object box

    return x;
}

void wombifier_free(t_wombifier *x) {
    if (x->dlyL) sysmem_freeptr(x->dlyL);
    if (x->dlyR) sysmem_freeptr(x->dlyR);
    fir_free(&x->firL);
    fir_free(&x->firR);
    dsp_free((t_pxobject *)x);
}

// --------------------------------- Assist -----------------------------------
void wombifier_assist(t_wombifier *x, void *b, long m, long a, char *s) {
    if (m == ASSIST_INLET) {
        switch (a) {
            case 0: sprintf(s, "Signal In L"); break;
            case 1: sprintf(s, "Signal In R"); break;
            case 2: sprintf(s, "Control In — attributes/messages (e.g. \"@cutoff 1000 @fir 1 @fir_prime 1\")"); break;
        }
    } else {
        switch (a) {
            case 0: sprintf(s, "Signal Out L"); break;
            case 1: sprintf(s, "Signal Out R"); break;
        }
    }
}

// --------------------------------- DSP --------------------------------------
static void fir_rebuild_if_needed(t_wombifier *x) {
    if (!x->use_fir) {
        x->fir_dirty = 0;
        return;
    }

    if (!(x->fir_dirty || !x->firL.taps || !x->firR.taps ||
          x->firL.ntaps != x->fir_ntaps || x->firR.ntaps != x->fir_ntaps)) {
        return;
    }

    fir_prepare(&x->firL, (int)x->fir_ntaps);
    fir_prepare(&x->firR, (int)x->fir_ntaps);
    fir_make_lowpass(&x->firL, x->sr, x->cutoff, (int)x->fir_prime, x->fir_asym, (int)x->fir_energy_norm);
    fir_make_lowpass(&x->firR, x->sr, x->cutoff, (int)x->fir_prime, x->fir_asym, (int)x->fir_energy_norm);
    x->fir_dirty = 0;
}

void wombifier_dsp64(t_wombifier *x, t_object *dsp64, short *count, double samplerate,
                     long maxvectorsize, long flags) {
    double prev_sr = x->sr;
    if (count) {
        x->in_connected[0] = (char)(count[0] != 0);
        x->in_connected[1] = (char)(count[1] != 0);
    } else {
        x->in_connected[0] = x->in_connected[1] = 0;
    }
    if (samplerate > 0 && samplerate != x->sr) {
        x->sr = samplerate;
        x->noise_g = 1.0 - exp(-2.0 * M_PI * (300.0 / x->sr));
        biquad_set_lp(&x->noise_filter[0], x->sr, 600.0, 0.85);
        biquad_set_lp(&x->noise_filter[1], x->sr, 300.0, 0.75);
        resonator_init(&x->resL, x->sr);
        resonator_init(&x->resR, x->sr);
        biquad_set_womb_response(&x->mtf_filter[0], x->sr);
        biquad_set_womb_response(&x->mtf_filter[1], x->sr);
        biquad_set_maternal_voice(&x->voice_enhance[0], x->sr);
        biquad_set_maternal_voice(&x->voice_enhance[1], x->sr);
        x->coeffs_dirty = 1;
        x->fir_dirty = 1;
    } else if (samplerate > 0) {
        x->sr = samplerate;
    } else if (x->sr <= 0.0) {
        x->sr = (prev_sr > 0.0 ? prev_sr : 48000.0);
    }

    ensure_delay(x);
    if (x->coeffs_dirty) update_coeffs(x, 0.0);
    fir_rebuild_if_needed(x);

    object_method(dsp64, gensym("dsp_add64"), x, wombifier_perform64, 0, NULL);
}

// -------------------------------- Perform -----------------------------------
void wombifier_perform64(t_wombifier *x, t_object *dsp64, double **ins, long numins,
                         double **outs, long numouts, long n, long flags, void *userparam) {

    double *inL = ins[0];
    double *inR = ins[1];
    double *outL = outs[0];
    double *outR = outs[1];

    double sr = x->sr;

    double phase = x->phase;
    double ph_inc = (x->bpm / 60.0) / sr; // cycles per sample

    // HRV / respiration update (lento)
    x->hrv_phase += x->hrv_rate / sr; if (x->hrv_phase >= 1.0) x->hrv_phase -= 1.0;
    x->resp_phase+= x->resp_rate / sr; if (x->resp_phase>= 1.0) x->resp_phase-= 1.0;
    double hrv_mod  = 0.98 + 0.04 * sin(2.0 * M_PI * x->hrv_phase);  // ±2%
    double resp_mod = 0.95 + 0.10 * sin(2.0 * M_PI * x->resp_phase); // ±5%
    ph_inc *= hrv_mod;

    double depth = clampd(x->depth, 0.0, 1.0);
    double wet   = clampd(x->wet,   0.0, 1.0);
    double noise_amt = clampd(x->noise_amt, 0.0, 1.0);
    double cutmod = clampd(x->cutmod, 0.0, 1.0);
    double soft = clampd(x->soft, 0.0, 1.0);
    double width = clampd(x->width, 0.0, 1.0);

    double water = clampd(x->water, 0.0, 1.0);
    double base_ms = clampd(x->water_ms, 1.0, 30.0);
    double rate = clampd(x->water_rate, 0.01, 5.0);
    double stereo_ms = clampd(x->stereo_ms, 0.0, 20.0) * width;

    double lfo_inc = rate / sr;
    double lfo_incR = lfo_inc * (1.0 - 0.03 * width); // 0..3% di detune
    double lfoL = x->lfoL, lfoR = x->lfoR;

    long size = x->dly_size;
    long wr = x->wr_idx;

    double base_samps   = (base_ms   * 0.001) * sr;
    double stereo_samps = (stereo_ms * 0.001) * sr;
    double feedback = clampd(x->feedback, 0.0, 0.3);

    double cutoff_s = x->cutoff_smooth;

    int use_fir = (x->use_fir != 0);
    if (use_fir && x->fir_dirty) {
        // rarissimo, ma al bisogno
        fir_rebuild_if_needed(x);
    }

    for (long i = 0; i < n; ++i) {
        // heartbeat
        if (phase >= 1.0) phase -= 1.0;
        double env = heart_env_area_norm(x->gauss1_mu, x->gauss1_sigma,
                                         x->gauss2_mu, x->gauss2_sigma, x->gauss2_gain,
                                         phase);
        phase += ph_inc;

        // modulation (media ~1.0)
        double mod = (1.0 - depth) + depth * env;

        // input
        double l = inL ? inL[i] : 0.0;
        double r = (x->in_connected[1] && inR) ? inR[i] : l;

        if (x->mono) {
            double m = 0.5 * (l + r);
            l = r = m;
        }

        // noise (ben filtrato)
        double wn = rand_uniform_pm1(&x->rng);
        wn = biquad_process(&x->noise_filter[0], wn);
        wn = biquad_process(&x->noise_filter[1], wn);
        x->noise_lp += x->noise_g * (wn - x->noise_lp);
        double namp = noise_amt * (0.35 + 0.65 * env);
        double nval = x->noise_lp * namp;

        // somma mod + noise prima dell'occlusione
        double sL = l * mod + nval;
        double sR = r * mod + nval;

        // Occlusione: FIR (linear-phase, prime-mode) oppure IIR LP dinamico
        if (use_fir) {
            // FIR statico sul cutoff corrente (perfetto e naturale)
            sL = fir_process_1(&x->firL, sL);
            sR = fir_process_1(&x->firR, sR);
        } else {
            // IIR: cutoff mod dal battito + smoothing + refresh periodico
            double target_cut = x->cutoff * (0.85 + 0.5 * cutmod * env);
            target_cut = clampd(target_cut, 40.0, sr * 0.45);
            cutoff_s += 0.005 * (target_cut - cutoff_s);
            if ((i & 15) == 0) update_coeffs(x, cutoff_s);
            sL = biquad_process(&x->lpfL, sL);
            sR = biquad_process(&x->lpfR, sR);
        }

        // Watery (modulated short delay/chorus) — simmetrico con width=0
        if (water > 1e-5) {
            double depth_samps = (0.25 + 0.75 * water) * 0.5 * base_samps;
            double stereo_offset = stereo_samps * 0.5;
            if (stereo_offset > base_samps - 1.0) stereo_offset = base_samps - 1.0;

            double delayL = base_samps - stereo_offset + sin(2.0 * M_PI * lfoL) * depth_samps;
            double delayR = base_samps + stereo_offset + sin(2.0 * M_PI * lfoR) * depth_samps;

            double tapL = tap_delay(x->dlyL, size, wr, delayL);
            double tapR = tap_delay(x->dlyR, size, wr, delayR);

            double mix = 0.35 + 0.65 * water;
            double wetL = (1.0 - mix) * sL + mix * tapL;
            double wetR = (1.0 - mix) * sR + mix * tapR;

            x->dlyL[wr] = sL + tapL * feedback;
            x->dlyR[wr] = sR + tapR * feedback;

            sL = wetL; sR = wetR;
        } else {
            x->dlyL[wr] = sL; x->dlyR[wr] = sR;
        }

        wr++; if (wr >= size) wr = 0;
        lfoL += lfo_inc;  if (lfoL >= 1.0)  lfoL -= 1.0;
        lfoR += lfo_incR; if (lfoR >= 1.0) lfoR -= 1.0;

        // Thickness (simmetrico)
        double thick = x->thickness * 0.7;
        sL *= (1.0 + thick * (1.0 - env));
        sR *= (1.0 + thick * (1.0 - env));

        // Softclip pulito (no oversampling spurio)
        if (soft > 0.0) {
            sL = softclip(sL, soft);
            sR = softclip(sR, soft);
        }

        // Envelope following (dinamica)
        double in_abs = 0.5 * fabs(l + r);
        x->env_follow = in_abs + x->env_smooth * (x->env_follow - in_abs);
        double dyn_mod = 1.0 + x->dynamics * (x->env_follow - 0.5);
        (void)dyn_mod; // attualmente lo usi per shaping complessivo, teniamolo neutro

        // Resonators (simmetrici)
        double res_amt = x->bodyres * (0.7 + 0.3 * env);
        sL += resonator_process(&x->resL, sL) * res_amt;
        sR += resonator_process(&x->resR, sR) * res_amt;

        // Warmth
        if (x->warmth > 0.0) {
            double warm = x->warmth * (0.5 + 0.5 * env);
            sL += warm * tanh(sL * 1.5) * 0.33;
            sR += warm * tanh(sR * 1.5) * 0.33;
        }

        // MTF / Voice opzionali
        if (x->mtf_on) { sL = biquad_process(&x->mtf_filter[0], sL); sR = biquad_process(&x->mtf_filter[1], sR); }
        if (x->mvoice_on) { sL = biquad_process(&x->voice_enhance[0], sL); sR = biquad_process(&x->voice_enhance[1], sR); }

        // Respirazione
        sL *= resp_mod; sR *= resp_mod;

        // Level monitoring + safety
        double inst_level = 0.5 * (sL*sL + sR*sR);
        x->leq_acc += inst_level; x->leq_count++;
        if (inst_level > x->peak_level) x->peak_level = inst_level;

        double safety_limit = 0.707; // -3 dB
        if (inst_level > safety_limit) {
            double scale = sqrt(safety_limit / inst_level);
            sL *= scale; sR *= scale;
        }

        // wet/dry + clamp
        double ol = sL * wet + l * (1.0 - wet);
        double orr= sR * wet + r * (1.0 - wet);
        if (ol >  1.0) ol =  1.0; else if (ol < -1.0) ol = -1.0;
        if (orr>  1.0) orr=  1.0; else if (orr< -1.0) orr= -1.0;

        outL[i] = ol;
        outR[i] = orr;
    }

    x->phase = phase;
    x->cutoff_smooth = cutoff_s;
    x->wr_idx = wr;
    x->lfoL = lfoL; x->lfoR = lfoR;
}

// --------------------------- Anything handler --------------------------------
void wombifier_anything(t_wombifier *x, t_symbol *s, long argc, t_atom *argv) {
    if (!s || !s->s_name) return;
    const char *name = s->s_name;
    if (name[0] == '@') {
        t_symbol *attr = gensym(name + 1);
        t_max_err err = object_attr_setvalueof((t_object *)x, attr, argc, argv);
        if (err != MAX_ERR_NONE) {
            object_error((t_object *)x, "unknown attribute '%s' or invalid value", attr->s_name);
        }
        return;
    }

    if (!strcmp(name, "cutoff") && argc)      { wombifier_set_cutoff(x, NULL, 1, argv); return; }
    if (!strcmp(name, "q") && argc)           { wombifier_set_q(x, NULL, 1, argv); return; }

    if (!strcmp(name, "fir") && argc)         { object_attr_setvalueof((t_object*)x, gensym("fir"), 1, argv); return; }
    if (!strcmp(name, "fir_prime") && argc)   { wombifier_set_fir_prime(x, NULL, 1, argv); return; }
    if (!strcmp(name, "fir_ntaps") && argc)   { wombifier_set_fir_ntaps(x, NULL, 1, argv); return; }
    if (!strcmp(name, "fir_asym") && argc)    { wombifier_set_fir_asym(x, NULL, 1, argv); return; }
    if (!strcmp(name, "fir_energy") && argc)  { wombifier_set_fir_energy(x, NULL, 1, argv); return; }

    if (!strcmp(name, "width") && argc)    { object_attr_setvalueof((t_object*)x, gensym("width"), 1, argv); return; }
    if (!strcmp(name, "water") && argc)    { object_attr_setvalueof((t_object*)x, gensym("water"), 1, argv); return; }
    if (!strcmp(name, "soft") && argc)     { object_attr_setvalueof((t_object*)x, gensym("soft"), 1, argv); return; }
    if (!strcmp(name, "bodyres") && argc)  { object_attr_setvalueof((t_object*)x, gensym("bodyres"), 1, argv); return; }
    if (!strcmp(name, "warmth") && argc)   { object_attr_setvalueof((t_object*)x, gensym("warmth"), 1, argv); return; }
    if (!strcmp(name, "water_ms") && argc) { object_attr_setvalueof((t_object*)x, gensym("water_ms"), 1, argv); return; }
    if (!strcmp(name, "water_rate") && argc) { object_attr_setvalueof((t_object*)x, gensym("water_rate"), 1, argv); return; }
    if (!strcmp(name, "stereo_ms") && argc)  { object_attr_setvalueof((t_object*)x, gensym("stereo_ms"), 1, argv); return; }
    if (!strcmp(name, "mono") && argc)       { object_attr_setvalueof((t_object*)x, gensym("mono"), 1, argv); return; }

    object_post((t_object *)x, "Unknown message '%s'", s->s_name);
}
