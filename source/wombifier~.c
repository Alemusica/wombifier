// wombifier~.c — Max/MSP external that makes incoming audio feel "in utero" with optional watery texture
// Author: You + ChatGPT
// Creator: Me
// License: MIT
// Build: place in a Max SDK MSP external project and compile as a 64-bit external

#include "ext.h"
#include "ext_obex.h"
#include "z_dsp.h"
#include "ext_sysmem.h"  // Changed from sysmemory.h
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ----------------------------- Biquad LPF (RBJ) -----------------------------
typedef struct _biquad {
    double b0, b1, b2, a1, a2;
    double x1, x2, y1, y2;
} t_biquad;

static void biquad_clear(t_biquad *b) {
    b->x1 = b->x2 = b->y1 = b->y2 = 0.0;
}

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

    b->b0 = b0 / a0;
    b->b1 = b1 / a0;
    b->b2 = b2 / a0;
    b->a1 = a1 / a0;
    b->a2 = a2 / a0;
}

static inline double biquad_process(t_biquad *b, double x) {
    double y = b->b0 * x + b->b1 * b->x1 + b->b2 * b->x2 - b->a1 * b->y1 - b->a2 * b->y2;
    b->x2 = b->x1; b->x1 = x;
    b->y2 = b->y1; b->y1 = y;
    return y;
}

// Add bandpass after biquad_set_lp:
static void biquad_set_bp(t_biquad *b, double sr, double fc, double q) {
    if (fc < 20.0) fc = 20.0;
    if (fc > sr * 0.45) fc = sr * 0.45;
    if (q < 0.1) q = 0.1;

    double w0 = 2.0 * M_PI * (fc / sr);
    double cosw0 = cos(w0);
    double sinw0 = sin(w0);
    double alpha = sinw0 / (2.0 * q);

    double b0 = q * alpha;     // = sin(w0)/2
    double b1 = 0.0;
    double b2 = -q * alpha;
    double a0 = 1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 = 1.0 - alpha;

    b->b0 = b0 / a0;
    b->b1 = b1 / a0;
    b->b2 = b2 / a0;
    b->a1 = a1 / a0;
    b->a2 = a2 / a0;

    biquad_clear(b);
}

// Add lowshelf for depth:
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

// ----------------------------- FIR PrimeFIR-style ----------------------------
typedef struct _fir {
    double *taps;
    int ntaps;
    double *buf;
    int idx;
    int prime_mode;
    double asym;
    double gain_l1;
    double gain_l2;
} t_fir;

static int is_prime_i(int n) {
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (int k = 3; k * k <= n; k += 2) {
        if (n % k == 0) return 0;
    }
    return 1;
}

static void fir_free(t_fir *f) {
    if (f->taps) sysmem_freeptr(f->taps);
    if (f->buf) sysmem_freeptr(f->buf);
    memset(f, 0, sizeof(*f));
}

static void fir_init(t_fir *f, int ntaps) {
    fir_free(f);
    f->ntaps = (ntaps % 2 == 0) ? (ntaps + 1) : ntaps;
    f->taps = (double *)sysmem_newptrclear(sizeof(double) * f->ntaps);
    f->buf = (double *)sysmem_newptrclear(sizeof(double) * f->ntaps);
    f->idx = 0;
}

static void fir_make_lowpass(t_fir *f, double sr, double fc, int prime_mode, double asym, int energy_norm) {
    if (!f->taps || !f->buf) return;

    int N = f->ntaps;
    int M = N - 1;
    double fc_n = fc / (sr * 0.5);
    if (fc_n < 0.001) fc_n = 0.001;
    if (fc_n > 0.999) fc_n = 0.999;

    double center = M * (0.5 - 0.25 * asym);

    const double a0 = 0.35875;
    const double a1 = 0.48829;
    const double a2 = 0.14128;
    const double a3 = 0.01168;

    double sumL1 = 0.0;
    double sumL2 = 0.0;

    for (int n = 0; n < N; ++n) {
        double k = (double)n - center;
        double x = M_PI * k * fc_n;

        double sinc = (fabs(x) < 1e-12) ? fc_n : sin(x) / (M_PI * k);
        double warg = (double)n / M;
        double w = a0 - a1 * cos(2.0 * M_PI * warg) + a2 * cos(4.0 * M_PI * warg) - a3 * cos(6.0 * M_PI * warg);

        double tap = sinc * w;
        if (prime_mode && n != (int)round(center)) {
            if (!is_prime_i(n)) tap = 0.0;
        }

        f->taps[n] = tap;
        sumL1 += tap;
        sumL2 += tap * tap;
    }

    if (energy_norm) {
        double norm = sqrt(sumL2);
        if (norm > 1e-15) {
            for (int n = 0; n < N; ++n) f->taps[n] /= norm;
        }
        f->gain_l1 = 0.0;
        f->gain_l2 = 1.0;
    } else {
        if (fabs(sumL1) < 1e-15) sumL1 = 1.0;
        for (int n = 0; n < N; ++n) f->taps[n] /= sumL1;
        f->gain_l1 = 1.0;
        f->gain_l2 = 0.0;
    }

    memset(f->buf, 0, sizeof(double) * N);
    f->idx = 0;
    f->prime_mode = prime_mode;
    f->asym = asym;
}

static inline double fir_process_1(t_fir *f, double x) {
    if (!f->taps || !f->buf) return x;
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

// -------------------------- Resonator Bank -----------------------------
typedef struct _resonator {
    t_biquad bands[3];  // 3 resonant bands
    double gains[3];    // gain per band
} t_resonator;

static void resonator_init(t_resonator *r, double sr) {
    // Research-based resonant frequencies for amniotic cavity
    double freqs[3] = {200.0, 400.0, 600.0};  // Based on literature
    double qs[3]    = {4.0,   3.5,   3.0};    // Sharper resonances
    double gains[3] = {1.0,   0.7,   0.5};    // Natural decay
    
    for(int i = 0; i < 3; i++) {
        biquad_set_bp(&r->bands[i], sr, freqs[i], qs[i]);
        r->gains[i] = gains[i];
    }
}

static double resonator_process(t_resonator *r, double in) {
    double out = 0.0;
    for(int i = 0; i < 3; i++) {
        out += biquad_process(&r->bands[i], in) * r->gains[i];
    }
    return out;
}

// ------------------------------ Object struct -------------------------------
typedef struct _wombifier {
    t_pxobject x_obj;
    // base params
    double bpm;        // 20..200 (default 72)
    double depth;      // 0..1   (AM depth, default 0.6)
    double cutoff;     // Hz     (default 350)
    double q;          // resonance/Q (default 1.2)
    double noise_amt;  // 0..1   (default 0.15)
    double wet;        // 0..1   (default 1.0)

    // extras
    double cutmod;     // 0..1   cutoff modulation by heartbeat (default 0.35)
    double soft;       // 0..1   soft saturator amount (default 0.2)

    // watery/chorus
    double water;      // 0..1   mix for watery texture (default 0.35)
    double water_ms;   // base delay ms (default 12)
    double water_rate; // Hz     LFO rate (default 0.25)
    double stereo_ms;  // static extra delay on R channel (default 10)
    double width;      // 0..1 stereo asymmetry (default 0)

    // new parameters
    double thickness; // 0..1 more body (default 0.3)
    double warmth;     // 0..1 subtle harmonics (default 0.3)
    double bodyres;    // 0..1 body resonance (default 0.4)
    double feedback;   // 0..0.3 water feedback
    double dynamics;   // 0..1 envelope following (default 0.25)

    // dsp state
    double sr;
    double phase;      // 0..1 heartbeat phase
    double gauss1_mu, gauss1_sigma;
    double gauss2_mu, gauss2_sigma, gauss2_gain;

    // noise
    unsigned int rng;
    double noise_lp;   // one-pole state
    double noise_g;    // one-pole g

    // filters per channel
    t_biquad lpfL;
    t_biquad lpfR;
    double cutoff_smooth; // runtime smoothed cutoff

    // watery delay buffers
    double *dlyL; double *dlyR;
    long dly_size;  // buffer length in samples
    long wr_idx;    // write index
    double lfoL, lfoR; // 0..1

    // better noise generation
    t_biquad noise_filter[2];

    // improved delay interpolation
    double dly_zm1[2]; // previous samples for interpolation

    // resonators
    t_resonator resL, resR;

    // envelope follower
    double env_follow;
    double env_smooth;

    char coeffs_dirty;

    // Research-based additions
    t_biquad mtf_filter[2];     // Maternal transfer function
    t_biquad voice_enhance[2];  // Maternal voice enhancement
    long use_mtf;
    long use_mvoice;

    // HRV and respiratory modulation
    double hrv_phase;           // 0..1 slow HRV modulation
    double resp_phase;          // 0..1 respiratory phase
    double hrv_rate;           // Hz, typically 0.1 Hz (10s period)
    double resp_rate;          // Hz, typically 0.25 Hz (4s period)

    // FIR linear-phase filtering
    t_fir firL, firR;
    long use_fir;
    long fir_ntaps;
    double fir_asym;
    long fir_prime;
    long fir_energy_norm;
    char fir_dirty;

    // Level monitoring
    double leq_acc;            // Accumulated energy for Leq
    double peak_level;         // Peak level tracking
    long leq_count;           // Sample count for Leq calculation
} t_wombifier;

// class pointer
t_class *wombifier_class;

// --------------------------------- Helpers ----------------------------------
static inline double clampd(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static inline double softclip(double x, double amt) { // amt 0..1
    if (amt <= 0.0) return x;
    double drive = 1.0 + amt * 12.0; // gentle -> stronger
    double y = tanh(drive * x);
    double norm = tanh(drive);
    return y / (norm > 0.0 ? norm : 1.0);
}

static inline double rand_uniform(t_wombifier *x) { // [-1,1]
    x->rng ^= x->rng << 13; x->rng ^= x->rng >> 17; x->rng ^= x->rng << 5; // xorshift32
    return ((x->rng * (1.0 / 2147483648.0)) - 1.0);
}

static double heart_env(t_wombifier *x, double phase) {
    double g1 = exp(-0.5 * pow((phase - x->gauss1_mu) / x->gauss1_sigma, 2.0));
    double g2 = exp(-0.5 * pow((phase - x->gauss2_mu) / x->gauss2_sigma, 2.0)) * x->gauss2_gain;
    double env = g1 + g2;

    double area = 2.5066282746310002 * (x->gauss1_sigma + x->gauss2_sigma * x->gauss2_gain);
    if (area > 1e-9) env /= area;

    env = env / (1.0 + 0.5 * env);
    if (env > 3.0) env = 3.0;
    return env;
}

static void update_coeffs(t_wombifier *x, double cutoff_override) {
    double c = cutoff_override > 0 ? cutoff_override : x->cutoff;
    biquad_set_lp(&x->lpfL, x->sr, c, x->q);
    biquad_set_lp(&x->lpfR, x->sr, c, x->q);
    x->coeffs_dirty = 0;
}

static void ensure_delay(t_wombifier *x) {
    // allocate up to 100 ms of delay at current SR
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
    // Catmull-Rom interpolation for smooth delay taps
    if (delay_samps < 1.0) delay_samps = 1.0;
    if (delay_samps > size - 4) delay_samps = size - 4;

    double read_pos = (double)wr_idx - delay_samps;
    while (read_pos < 0.0) read_pos += size;

    long i1 = (long)read_pos;
    long i0 = i1 - 1; if (i0 < 0) i0 += size;
    long i2 = i1 + 1; if (i2 >= size) i2 -= size;
    long i3 = i2 + 1; if (i3 >= size) i3 -= size;

    double frac = read_pos - (double)i1;
    double a0 = buf[i0];
    double a1 = buf[i1];
    double a2 = buf[i2];
    double a3 = buf[i3];

    double c0 = a1;
    double c1 = 0.5 * (a2 - a0);
    double c2 = a0 - 2.5 * a1 + 2.0 * a2 - 0.5 * a3;
    double c3 = -0.5 * a0 + 1.5 * a1 - 1.5 * a2 + 0.5 * a3;

    return ((c3 * frac + c2) * frac + c1) * frac + c0;
}

// ---------------------------- Method prototypes -----------------------------
void *wombifier_new(t_symbol *s, long argc, t_atom *argv);
void wombifier_free(t_wombifier *x);
void wombifier_assist(t_wombifier *x, void *b, long m, long a, char *s);

void wombifier_dsp64(t_wombifier *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags);
void wombifier_perform64(t_wombifier *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long sampleframes, long flags, void *userparam);

// Forward-declare research-based biquad helpers so they can be used in wombifier_new
static void biquad_set_womb_response(t_biquad *b, double sr);
static void biquad_set_maternal_voice(t_biquad *b, double sr);

// --- NEW: generic message handler to accept "@attr value" inlets ----------
void wombifier_anything(t_wombifier *x, t_symbol *s, long argc, t_atom *argv);

// attribute setters (need to refresh coeffs when cutoff/Q change)
void wombifier_set_cutoff(t_wombifier *x, void *attr, long ac, t_atom *av);
void wombifier_set_q(t_wombifier *x, void *attr, long ac, t_atom *av);
void wombifier_set_fir_enable(t_wombifier *x, void *attr, long ac, t_atom *av);
void wombifier_set_fir_ntaps(t_wombifier *x, void *attr, long ac, t_atom *av);
void wombifier_set_fir_asym(t_wombifier *x, void *attr, long ac, t_atom *av);
void wombifier_set_fir_prime(t_wombifier *x, void *attr, long ac, t_atom *av);
void wombifier_set_fir_energy(t_wombifier *x, void *attr, long ac, t_atom *av);

// ---------------------------------- Main ------------------------------------
C74_EXPORT void ext_main(void *r) {
    t_class *c = class_new("wombifier~", (method)wombifier_new, (method)wombifier_free, (long)sizeof(t_wombifier), 0L, A_GIMME, 0);

    post("wombifier~ — build %s %s", __DATE__, __TIME__);

    class_addmethod(c, (method)wombifier_assist, "assist", A_CANT, 0);
    class_addmethod(c, (method)wombifier_dsp64, "dsp64", A_CANT, 0);
    // --- NEW: register anything handler so messages like "@cutoff 100" are accepted ---
    class_addmethod(c, (method)wombifier_anything, "anything", A_GIMME, 0);

    class_dspinit(c);

    // base params
    CLASS_ATTR_DOUBLE(c, "bpm", 0, t_wombifier, bpm);
    CLASS_ATTR_LABEL(c,  "bpm", 0, "Heart Rate (BPM)");
    CLASS_ATTR_FILTER_CLIP(c, "bpm", 20.0, 200.0);

    CLASS_ATTR_DOUBLE(c, "depth", 0, t_wombifier, depth);
    CLASS_ATTR_LABEL(c,  "depth", 0, "Pulse Depth (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "depth", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "cutoff", 0, t_wombifier, cutoff);
    CLASS_ATTR_LABEL(c,  "cutoff", 0, "Low-Pass Cutoff (Hz)");
    CLASS_ATTR_ACCESSORS(c, "cutoff", NULL, wombifier_set_cutoff);
    CLASS_ATTR_FILTER_CLIP(c, "cutoff", 40.0, 4000.0);

    // Replace resonance attribute for Q with new name
    CLASS_ATTR_DOUBLE(c, "q", 0, t_wombifier, q);
    CLASS_ATTR_LABEL(c,  "q", 0, "Low-Pass Resonance (Q)");
    CLASS_ATTR_ACCESSORS(c, "q", NULL, wombifier_set_q);
    CLASS_ATTR_FILTER_CLIP(c, "q", 0.3, 8.0);

    CLASS_ATTR_DOUBLE(c, "noise", 0, t_wombifier, noise_amt);
    CLASS_ATTR_LABEL(c,  "noise", 0, "Fluid Noise (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "noise", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "wet", 0, t_wombifier, wet);
    CLASS_ATTR_LABEL(c,  "wet", 0, "Wet/Dry (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "wet", 0.0, 1.0);

    // extras
    CLASS_ATTR_DOUBLE(c, "cutmod", 0, t_wombifier, cutmod);
    CLASS_ATTR_LABEL(c,  "cutmod", 0, "Cutoff Mod by Heartbeat (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "cutmod", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "soft", 0, t_wombifier, soft);
    CLASS_ATTR_LABEL(c,  "soft", 0, "Soft Limiter Amount (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "soft", 0.0, 1.0);

    // watery
    CLASS_ATTR_DOUBLE(c, "water", 0, t_wombifier, water);
    CLASS_ATTR_LABEL(c,  "water", 0, "Watery Mix (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "water", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "water_ms", 0, t_wombifier, water_ms);
    CLASS_ATTR_LABEL(c,  "water_ms", 0, "Watery Base Delay (ms)");
    CLASS_ATTR_FILTER_CLIP(c, "water_ms", 1.0, 30.0);

    CLASS_ATTR_DOUBLE(c, "water_rate", 0, t_wombifier, water_rate);
    CLASS_ATTR_LABEL(c,  "water_rate", 0, "Watery LFO Rate (Hz)");
    CLASS_ATTR_FILTER_CLIP(c, "water_rate", 0.01, 5.0);

    CLASS_ATTR_DOUBLE(c, "stereo_ms", 0, t_wombifier, stereo_ms);
    CLASS_ATTR_LABEL(c,  "stereo_ms", 0, "Static Right Delay (ms)");
    CLASS_ATTR_FILTER_CLIP(c, "stereo_ms", 0.0, 20.0);

    CLASS_ATTR_DOUBLE(c, "width", 0, t_wombifier, width);
    CLASS_ATTR_LABEL(c,  "width", 0, "Stereo Width (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "width", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "thickness", 0, t_wombifier, thickness);
    CLASS_ATTR_LABEL(c, "thickness", 0, "Body Thickness (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "thickness", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "warmth", 0, t_wombifier, warmth);
    CLASS_ATTR_LABEL(c, "warmth", 0, "Harmonic Warmth (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "warmth", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "bodyres", 0, t_wombifier, bodyres);
    CLASS_ATTR_LABEL(c, "bodyres", 0, "Body Resonance (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "bodyres", 0.0, 1.0);

    CLASS_ATTR_DOUBLE(c, "feedback", 0, t_wombifier, feedback);
    CLASS_ATTR_LABEL(c, "feedback", 0, "Water Feedback (0-0.3)");
    CLASS_ATTR_FILTER_CLIP(c, "feedback", 0.0, 0.3);

    CLASS_ATTR_DOUBLE(c, "dynamics", 0, t_wombifier, dynamics);
    CLASS_ATTR_LABEL(c, "dynamics", 0, "Dynamic Response (0-1)");
    CLASS_ATTR_FILTER_CLIP(c, "dynamics", 0.0, 1.0);

    CLASS_ATTR_LONG(c, "mtf", 0, t_wombifier, use_mtf);
    CLASS_ATTR_LABEL(c, "mtf", 0, "Enable Maternal Transfer (0/1)");
    CLASS_ATTR_FILTER_CLIP(c, "mtf", 0, 1);

    CLASS_ATTR_LONG(c, "mvoice", 0, t_wombifier, use_mvoice);
    CLASS_ATTR_LABEL(c, "mvoice", 0, "Enable Maternal Voice Enhance (0/1)");
    CLASS_ATTR_FILTER_CLIP(c, "mvoice", 0, 1);

    CLASS_ATTR_LONG(c, "fir", 0, t_wombifier, use_fir);
    CLASS_ATTR_LABEL(c, "fir", 0, "Enable Linear-Phase FIR (0/1)");
    CLASS_ATTR_ACCESSORS(c, "fir", NULL, wombifier_set_fir_enable);
    CLASS_ATTR_FILTER_CLIP(c, "fir", 0, 1);

    CLASS_ATTR_LONG(c, "fir_ntaps", 0, t_wombifier, fir_ntaps);
    CLASS_ATTR_LABEL(c, "fir_ntaps", 0, "FIR Taps (odd 129-4097)");
    CLASS_ATTR_ACCESSORS(c, "fir_ntaps", NULL, wombifier_set_fir_ntaps);
    CLASS_ATTR_FILTER_CLIP(c, "fir_ntaps", 129, 4097);

    CLASS_ATTR_DOUBLE(c, "fir_asym", 0, t_wombifier, fir_asym);
    CLASS_ATTR_LABEL(c, "fir_asym", 0, "FIR Asymmetry (0-1)");
    CLASS_ATTR_ACCESSORS(c, "fir_asym", NULL, wombifier_set_fir_asym);
    CLASS_ATTR_FILTER_CLIP(c, "fir_asym", 0.0, 1.0);

    CLASS_ATTR_LONG(c, "fir_prime", 0, t_wombifier, fir_prime);
    CLASS_ATTR_LABEL(c, "fir_prime", 0, "FIR Prime Mode (0/1)");
    CLASS_ATTR_ACCESSORS(c, "fir_prime", NULL, wombifier_set_fir_prime);
    CLASS_ATTR_FILTER_CLIP(c, "fir_prime", 0, 1);

    CLASS_ATTR_LONG(c, "fir_energy", 0, t_wombifier, fir_energy_norm);
    CLASS_ATTR_LABEL(c, "fir_energy", 0, "FIR Energy Normalization (0/1)");
    CLASS_ATTR_ACCESSORS(c, "fir_energy", NULL, wombifier_set_fir_energy);
    CLASS_ATTR_FILTER_CLIP(c, "fir_energy", 0, 1);

    class_register(CLASS_BOX, c);
    wombifier_class = c;
}

// --------------------------------- New/Free ---------------------------------
void *wombifier_new(t_symbol *s, long argc, t_atom *argv) {
    t_wombifier *x = (t_wombifier *)object_alloc(wombifier_class);
    if (!x) return NULL;

    dsp_setup((t_pxobject *)x, 2);      // 2 signal inlets (L/R)
    outlet_new((t_object *)x, "signal"); // 2 signal outlets (L/R)
    outlet_new((t_object *)x, "signal");

    // NEW: add a dedicated control inlet for attribute messages (e.g. "@cutoff 120")
    // Note: signal inlets created by dsp_setup do not accept control messages,
    // so we create a non-signal inlet that will receive "anything".
    inlet_new((t_object *)x, NULL);
    
    x->sr        = sys_getsr(); if (x->sr <= 0) x->sr = 48000.0;
    x->bpm       = 72.0;
    x->depth     = 0.6;
    x->cutoff    = 350.0;
    x->q         = 1.2;
    x->noise_amt = 0.15;
    x->wet       = 1.0;

    x->cutmod    = 0.35;
    x->soft      = 0.2;

    x->water     = 0.35;
    x->water_ms  = 12.0;
    x->water_rate= 0.25;
    x->stereo_ms = 0.0;
    x->width     = 0.0;

    x->thickness = 0.3;
    x->warmth = 0.3;
    x->bodyres = 0.4;
    x->feedback = 0.15;
    x->dynamics = 0.25;
    x->env_follow = 0.0;
    x->env_smooth = 0.99; // smoothing coefficient

    x->phase = 0.0;
    x->gauss1_mu = 0.05; x->gauss1_sigma = 0.04;
    x->gauss2_mu = 0.32; x->gauss2_sigma = 0.05; x->gauss2_gain = 0.7;

    x->rng = 222222227u; // deterministic seed

    x->noise_lp = 0.0;
    double fc_noise = 300.0; // Hz
    x->noise_g = 1.0 - exp(-2.0 * M_PI * (fc_noise / x->sr));

    biquad_clear(&x->lpfL);
    biquad_clear(&x->lpfR);
    update_coeffs(x, 0.0);
    x->cutoff_smooth = x->cutoff;

    // Initialize noise filters
    biquad_set_lp(&x->noise_filter[0], x->sr, 600.0, 0.85);
    biquad_set_lp(&x->noise_filter[1], x->sr, 300.0, 0.75);

    // Initialize resonators
    resonator_init(&x->resL, x->sr);
    resonator_init(&x->resR, x->sr);

    // Initialize research-based filters
    biquad_set_womb_response(&x->mtf_filter[0], x->sr);
    biquad_set_womb_response(&x->mtf_filter[1], x->sr);
    biquad_set_maternal_voice(&x->voice_enhance[0], x->sr);
    biquad_set_maternal_voice(&x->voice_enhance[1], x->sr);
    x->use_mtf = 0;
    x->use_mvoice = 0;
    
    // Initialize modulation rates
    x->hrv_rate = 0.1;   // 0.1 Hz for HRV
    x->resp_rate = 0.25; // 0.25 Hz for respiration
    x->hrv_phase = 0.0;
    x->resp_phase = 0.0;
    
    // Initialize level monitoring
    x->leq_acc = 0.0;
    x->peak_level = 0.0;
    x->leq_count = 0;
    
    x->dlyL = x->dlyR = NULL; x->dly_size = 0; x->wr_idx = 0;
    x->lfoL = 0.0; x->lfoR = 0.0;

    memset(&x->firL, 0, sizeof(t_fir));
    memset(&x->firR, 0, sizeof(t_fir));
    x->use_fir = 0;
    x->fir_ntaps = 513;
    x->fir_asym = 0.3;
    x->fir_prime = 0;
    x->fir_energy_norm = 1;
    x->fir_dirty = 1;

    ensure_delay(x);

    attr_args_process(x, (short)argc, argv); // apply @attrs from the object box

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
            case 2: sprintf(s, "Control In — attributes/messages (e.g. \"@cutoff 120\")"); break;
        }
    } else {
        switch (a) {
            case 0: sprintf(s, "Signal Out L"); break;
            case 1: sprintf(s, "Signal Out R"); break;
        }
    }
}

// --------------------------------- DSP --------------------------------------
void wombifier_dsp64(t_wombifier *x, t_object *dsp64, short *count, double samplerate, long maxvectorsize, long flags) {
    double prev_sr = x->sr;
    if (samplerate > 0) {
        if (samplerate != x->sr) {
            x->sr = samplerate;
            resonator_init(&x->resL, x->sr);
            resonator_init(&x->resR, x->sr);
            biquad_set_lp(&x->noise_filter[0], x->sr, 600.0, 0.85);
            biquad_set_lp(&x->noise_filter[1], x->sr, 300.0, 0.75);
            biquad_set_womb_response(&x->mtf_filter[0], x->sr);
            biquad_set_womb_response(&x->mtf_filter[1], x->sr);
            biquad_set_maternal_voice(&x->voice_enhance[0], x->sr);
            biquad_set_maternal_voice(&x->voice_enhance[1], x->sr);
            x->coeffs_dirty = 1;
            x->fir_dirty = 1;
        } else {
            x->sr = samplerate;
        }
    }

    if (x->sr <= 0.0) x->sr = prev_sr > 0.0 ? prev_sr : 48000.0;

    x->noise_g = 1.0 - exp(-2.0 * M_PI * (300.0 / x->sr));
    ensure_delay(x);
    if (x->use_fir) {
        if (!x->firL.taps || x->firL.ntaps != x->fir_ntaps) fir_init(&x->firL, (int)x->fir_ntaps);
        if (!x->firR.taps || x->firR.ntaps != x->fir_ntaps) fir_init(&x->firR, (int)x->fir_ntaps);
        if (x->firL.taps && (x->fir_dirty || x->firL.prime_mode != x->fir_prime || x->firL.asym != x->fir_asym)) {
            fir_make_lowpass(&x->firL, x->sr, x->cutoff, (int)x->fir_prime, x->fir_asym, (int)x->fir_energy_norm);
        }
        if (x->firR.taps && (x->fir_dirty || x->firR.prime_mode != x->fir_prime || x->firR.asym != x->fir_asym)) {
            fir_make_lowpass(&x->firR, x->sr, x->cutoff, (int)x->fir_prime, x->fir_asym, (int)x->fir_energy_norm);
        }
        x->fir_dirty = 0;
    } else {
        if (x->firL.taps) fir_free(&x->firL);
        if (x->firR.taps) fir_free(&x->firR);
    }
    if (x->coeffs_dirty) update_coeffs(x, 0.0);
    object_method(dsp64, gensym("dsp_add64"), x, wombifier_perform64, 0, NULL);
}

static void biquad_set_womb_response(t_biquad *b, double sr) {
    // Research-based MTF (womb transfer function)
    // Strong attenuation above 1kHz (-20 to -30dB/oct)
    double fc = 1000.0;
    double gain = -24.0; // dB
    biquad_set_lowshelf(b, sr, fc, gain);
}

static void biquad_set_maternal_voice(t_biquad *b, double sr) {
    // Formant emphasis filter for maternal voice (~500Hz boost)
    biquad_set_lowshelf(b, sr, 500.0, 3.0); // +3dB boost for lower formants
}

void wombifier_perform64(t_wombifier *x, t_object *dsp64, double **ins, long numins, double **outs, long numouts, long n, long flags, void *userparam) {
    double *inL = ins[0];
    double *inR = ins[1];
    double *outL = outs[0];
    double *outR = outs[1];

    double sr = x->sr;
    double phase = x->phase;
    double ph_inc = (x->bpm / 60.0) / sr; // cycles per sample

    double depth = clampd(x->depth, 0.0, 1.0);
    double wet   = clampd(x->wet,   0.0, 1.0);
    double noise_amt = clampd(x->noise_amt, 0.0, 1.0);
    double cutmod = clampd(x->cutmod, 0.0, 1.0);
    double soft = clampd(x->soft, 0.0, 1.0);

    // watery params
    double water = clampd(x->water, 0.0, 1.0);
    double base_ms = clampd(x->water_ms, 1.0, 30.0);
    double rate = clampd(x->water_rate, 0.01, 5.0);
    double width = clampd(x->width, 0.0, 1.0);
    double stereo_ms = clampd(x->stereo_ms, 0.0, 20.0) * width;
    double lfo_inc = rate / sr; // cycles/sample
    double lfo_incR = lfo_inc * (1.0 - 0.03 * width);

    long size = x->dly_size;
    long wr = x->wr_idx;
    double lfoL = x->lfoL;
    double lfoR = x->lfoR;

    // precompute base delays in samples
    double base_samps = (base_ms * 0.001) * sr;
    double stereo_samps = (stereo_ms * 0.001) * sr;
    double feedback = clampd(x->feedback, 0.0, 0.3);

    // smoothing for dynamic cutoff
    double cutoff_s = x->cutoff_smooth;

    // Add respiratory modulation calculation at the start of the processing loop
    // Update HRV and respiratory modulation
    x->hrv_phase += x->hrv_rate / sr;
    if(x->hrv_phase >= 1.0) x->hrv_phase -= 1.0;
    x->resp_phase += x->resp_rate / sr;
    if(x->resp_phase >= 1.0) x->resp_phase -= 1.0;
    
    // Compute modulation factors
    double hrv_mod = 0.98 + 0.04 * sin(2.0 * M_PI * x->hrv_phase);  // ±2% HR variation
    double resp_mod = 0.95 + 0.1 * sin(2.0 * M_PI * x->resp_phase); // ±5% amplitude
    
    // Modify heartbeat timing with HRV
    ph_inc *= hrv_mod;

    for (long i = 0; i < n; ++i) {
        // heartbeat envelope
        if (phase >= 1.0) phase -= 1.0;
        double env = heart_env(x, phase);
        phase += ph_inc;

        // amplitude modulation (1-depth) .. 1 with heartbeat
        double mod = (1.0 - depth) + depth * env;

        // input
        double l = inL ? inL[i] : 0.0;
        double r = inR ? inR[i] : 0.0;

        // Better noise generation
        double wn = rand_uniform(x);
        wn = biquad_process(&x->noise_filter[0], wn);
        wn = biquad_process(&x->noise_filter[1], wn);
        x->noise_lp += x->noise_g * (wn - x->noise_lp);

        // add fluid/blood noise (lowpassed, slightly tied to heartbeat)
        double namp = noise_amt * (0.35 + 0.65 * env);
        double nval = x->noise_lp * namp;

        // cutoff modulation by heartbeat (gentle, smoothed)
        double target_cut = x->cutoff * (0.85 + 0.5 * cutmod * env); // ~0.85x .. 1.35x
        target_cut = clampd(target_cut, 40.0, sr * 0.45);
        // smooth toward target
        cutoff_s += 0.005 * (target_cut - cutoff_s);
        // refresh coeffs occasionally (every 16 samples)
        if (!x->use_fir && (i & 15) == 0) update_coeffs(x, cutoff_s);

        double mod_inL = l * mod + nval;
        double mod_inR = r * mod + nval;
        double wl;
        double wrs;
        if (x->use_fir) {
            wl = fir_process_1(&x->firL, mod_inL);
            wrs = fir_process_1(&x->firR, mod_inR);
        } else {
            wl = biquad_process(&x->lpfL, mod_inL);
            wrs = biquad_process(&x->lpfR, mod_inR);
        }

        // --- Watery (modulated short delay/chorus) ---
        if (water > 0.0001) {
            double depth_samps = (0.25 + 0.75 * water) * 0.5 * base_samps; // up to ~0.5*base depth
            double stereo_offset = stereo_samps * 0.5; // keep average delay centred
            if (stereo_offset > base_samps - 1.0)
                stereo_offset = base_samps - 1.0; // avoid collapsing left delay
            double delayL = base_samps - stereo_offset + sin(2.0 * M_PI * lfoL) * depth_samps;
            double delayR = base_samps + stereo_offset + sin(2.0 * M_PI * lfoR) * depth_samps;

            double tapL = tap_delay(x->dlyL, size, wr, delayL);
            double tapR = tap_delay(x->dlyR, size, wr, delayR);

            double mix = 0.35 + 0.65 * water; // 0.35..1.0
            double wetL = (1.0 - mix) * wl + mix * tapL;
            double wetR = (1.0 - mix) * wrs + mix * tapR;

            x->dlyL[wr] = wl + tapL * feedback;
            x->dlyR[wr] = wrs + tapR * feedback;

            wl = wetL;
            wrs = wetR;
        } else {
            x->dlyL[wr] = wl;
            x->dlyR[wr] = wrs;
        }

        wr++; if (wr >= size) wr = 0;

        lfoL += lfo_inc; if (lfoL >= 1.0) lfoL -= 1.0;
        lfoR += lfo_incR; if (lfoR >= 1.0) lfoR -= 1.0;

        // Add thickness
        double thick = x->thickness * 0.7;
        wl = wl * (1.0 + thick * (1.0 - env));
        wrs = wrs * (1.0 + thick * (1.0 - env));

        if (soft > 0.0) {
            wl = softclip(wl, soft);
            wrs = softclip(wrs, soft);
        }

        // Dynamic envelope following
        double in_abs = fabs(l + r) * 0.5;
        x->env_follow = in_abs + x->env_smooth * (x->env_follow - in_abs);
        double dyn_mod = 1.0 + x->dynamics * (x->env_follow - 0.5);

        // Enhance modulation with dynamics
        mod *= (1.0 + 0.2 * dyn_mod);

        // Process through resonators
        double res_amtL = x->bodyres * (0.7 + 0.3 * env);
        double res_amtR = res_amtL;

        wl = wl + resonator_process(&x->resL, wl) * res_amtL;
        wrs = wrs + resonator_process(&x->resR, wrs) * res_amtR;

        // Add warmth with subtle saturation
        if (x->warmth > 0.0) {
            double warm = x->warmth * (0.5 + 0.5 * env);
            wl = wl + warm * tanh(wl * 1.5) * 0.33;
            wrs = wrs + warm * tanh(wrs * 1.5) * 0.33;
        }

        // Apply research-based filtering
        if (x->use_mtf) {
            wl = biquad_process(&x->mtf_filter[0], wl);
            wrs = biquad_process(&x->mtf_filter[1], wrs);
        }
        if (x->use_mvoice) {
            wl = biquad_process(&x->voice_enhance[0], wl);
            wrs = biquad_process(&x->voice_enhance[1], wrs);
        }

        // Apply respiratory modulation
        wl *= resp_mod;
        wrs *= resp_mod;

        // Level monitoring (dB calculation)
        double inst_level = (wl * wl + wrs * wrs) * 0.5;
        x->leq_acc += inst_level;
        x->leq_count++;
        
        if(inst_level > x->peak_level) x->peak_level = inst_level;
        
        // Safety limiting (45 dB Leq, 65 dB peak targets)
        double safety_limit = 0.707; // -3dB
        if(inst_level > safety_limit) {
            double scale = sqrt(safety_limit / inst_level);
            wl *= scale;
            wrs *= scale;
        }

        // wet/dry
        double ol = wl * wet + l * (1.0 - wet);
        double orr = wrs * wet + r * (1.0 - wet);

        // safety clamp
        if (ol > 1.0) ol = 1.0; else if (ol < -1.0) ol = -1.0;
        if (orr > 1.0) orr = 1.0; else if (orr < -1.0) orr = -1.0;

        outL[i] = ol;
        outR[i] = orr;
    }

    x->phase = phase;
    x->cutoff_smooth = cutoff_s;
    x->wr_idx = wr;
    x->lfoL = lfoL; x->lfoR = lfoR;
}

// --------------------------- NEW: anything handler ---------------------------
void wombifier_anything(t_wombifier *x, t_symbol *s, long argc, t_atom *argv)
{
    if (!s || !s->s_name) return;

    const char *name = s->s_name;
    // handle messages that start with '@' by forwarding to attribute system
    if (name[0] == '@') {
        // gensym on name+1 creates symbol without leading '@'
        t_symbol *attr = gensym(name + 1);
        // object_attr_setvalueof expects (t_object*, t_symbol*, argc, argv)
        t_max_err err = object_attr_setvalueof((t_object *)x, attr, argc, argv);
        if (err != MAX_ERR_NONE) {
            object_error((t_object *)x, "unknown attribute '%s' or invalid value", attr->s_name);
        }
        return;
    }

    // fallback: unknown message
    object_post((t_object *)x, "Unknown message '%s'", s->s_name);
}

// --------------------------- Attribute setters ------------------------------
void wombifier_set_cutoff(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        double v = atom_getfloat(av);
        x->cutoff = clampd(v, 40.0, 4000.0);
        x->cutoff_smooth = x->cutoff; // reset smooth target
        x->coeffs_dirty = 1;
        x->fir_dirty = 1;
    }
}

void wombifier_set_q(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        double v = atom_getfloat(av);
        x->q = clampd(v, 0.3, 8.0);
        x->coeffs_dirty = 1;
    }
}

void wombifier_set_fir_enable(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        long v = atom_getlong(av);
        x->use_fir = v ? 1 : 0;
        x->fir_dirty = 1;
    }
}

void wombifier_set_fir_ntaps(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        long n = atom_getlong(av);
        if (n < 129) n = 129;
        if (n > 4097) n = 4097;
        if ((n & 1) == 0) {
            if (n < 4097) n += 1;
            else n -= 1;
        }
        x->fir_ntaps = n;
        x->fir_dirty = 1;
    }
}

void wombifier_set_fir_asym(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        double v = atom_getfloat(av);
        x->fir_asym = clampd(v, 0.0, 1.0);
        x->fir_dirty = 1;
    }
}

void wombifier_set_fir_prime(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        long v = atom_getlong(av);
        x->fir_prime = v ? 1 : 0;
        x->fir_dirty = 1;
    }
}

void wombifier_set_fir_energy(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        long v = atom_getlong(av);
        x->fir_energy_norm = v ? 1 : 0;
        x->fir_dirty = 1;
    }
}
