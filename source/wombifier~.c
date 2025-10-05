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

    double b0 =  sinw0 * 0.5 * q;
    double b1 =  0.0;
    double b2 = -sinw0 * 0.5 * q;
    double a0 =  1.0 + alpha;
    double a1 = -2.0 * cosw0;
    double a2 =  1.0 - alpha;

    b->b0 = b0 / a0;
    b->b1 = b1 / a0;
    b->b2 = b2 / a0;
    b->a1 = a1 / a0;
    b->a2 = a2 / a0;
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

// ---------------------------- Oversampling Helper ----------------------------
typedef struct _oversampler {
    t_biquad up[2];   // 2x upsampling filters
    t_biquad down[2]; // 2x downsampling filters
    double z1, z2;    // holding samples
} t_oversampler;

static void oversampler_init(t_oversampler *o, double sr) {
    // Higher quality anti-aliasing filters
    biquad_set_lp(&o->up[0], sr*2, sr*0.45, 0.85);
    biquad_set_lp(&o->up[1], sr*2, sr*0.45, 0.85);
    biquad_set_lp(&o->down[0], sr*2, sr*0.45, 0.85);
    biquad_set_lp(&o->down[1], sr*2, sr*0.45, 0.85);
    o->z1 = o->z2 = 0.0;
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

    // oversampling for clipper
    t_oversampler osL, osR;

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
    
    // HRV and respiratory modulation
    double hrv_phase;           // 0..1 slow HRV modulation
    double resp_phase;          // 0..1 respiratory phase
    double hrv_rate;           // Hz, typically 0.1 Hz (10s period)
    double resp_rate;          // Hz, typically 0.25 Hz (4s period)
    
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

static inline double fast_pow3(double x) { // pow(x,3) but faster
    return x * x * x;
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
    // Two Gaussian pulses per beat (lub-dub) — phase in [0,1)
    double e1 = exp(-0.5 * pow((phase - x->gauss1_mu) / x->gauss1_sigma, 2.0));
    double e2 = exp(-0.5 * pow((phase - x->gauss2_mu) / x->gauss2_sigma, 2.0)) * x->gauss2_gain;
    double env = e1 + e2;            // ~0..1.7
    if (env > 1.0) env = 1.0;        // soft-limit
    return fast_pow3(env);           // accentuate peaks, calm the lows
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

static inline double tap_delay(double *buf, long size, long wr_idx, double delay_samps) {
    // Improved interpolation for smoother delay
    if (delay_samps < 1.0) delay_samps = 1.0;
    if (delay_samps > size - 4) delay_samps = size - 4;
    double read_pos = (double)wr_idx - delay_samps;
    while (read_pos < 0.0) read_pos += size;

    long i0 = (long)read_pos;
    long i1 = i0 + 1; if (i1 >= size) i1 -= size;
    long i2 = i1 + 1; if (i2 >= size) i2 -= size;
    
    double frac = read_pos - (double)i0;
    double a = buf[i0];
    double b = buf[i1];
    double c = buf[i2];
    
    // Cubic interpolation
    double t = frac;
    double t2 = t * t;
    double t3 = t2 * t;
    return a + 0.5 * (b - a) * t + (c - b - (b - a)) * t2 + ((b - a) - (c - b)) * t3;
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
    x->stereo_ms = 10.0;

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

    // Higher quality oversampling
    oversampler_init(&x->osL, x->sr);
    oversampler_init(&x->osR, x->sr);

    // Initialize resonators
    resonator_init(&x->resL, x->sr);
    resonator_init(&x->resR, x->sr);

    // Initialize research-based filters
    biquad_set_womb_response(&x->mtf_filter[0], x->sr);
    biquad_set_womb_response(&x->mtf_filter[1], x->sr);
    biquad_set_maternal_voice(&x->voice_enhance[0], x->sr);
    biquad_set_maternal_voice(&x->voice_enhance[1], x->sr);
    
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
    x->lfoL = 0.0; x->lfoR = 0.33; // small phase offset for width
    ensure_delay(x);

    attr_args_process(x, (short)argc, argv); // apply @attrs from the object box

    return x;
}

void wombifier_free(t_wombifier *x) {
    if (x->dlyL) sysmem_freeptr(x->dlyL);
    if (x->dlyR) sysmem_freeptr(x->dlyR);
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
    if (samplerate > 0) x->sr = samplerate;
    x->noise_g = 1.0 - exp(-2.0 * M_PI * (300.0 / x->sr));
    ensure_delay(x);
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
    double stereo_ms = clampd(x->stereo_ms, 0.0, 20.0);
    double lfo_inc = rate / sr; // cycles/sample

    long size = x->dly_size;
    long wr = x->wr_idx;
    double lfoL = x->lfoL;
    double lfoR = x->lfoR;

    // precompute base delays in samples
    double base_samps = (base_ms * 0.001) * sr;
    double stereo_samps = (stereo_ms * 0.001) * sr;

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
        if ((i & 15) == 0) update_coeffs(x, cutoff_s);

        // apply low-pass occlusion to modulated signal
        double wl = biquad_process(&x->lpfL, l * mod + nval);
        double wrs= biquad_process(&x->lpfR, r * mod + nval);

        // --- Watery (modulated short delay/chorus) ---
        double dl = wl, dr = wrs;
        if (water > 0.0001) {
            double depth_samps = (0.25 + 0.75 * water) * 0.5 * base_samps; // up to ~0.5*base depth
            double delayL = base_samps + sin(2.0 * M_PI * lfoL) * depth_samps;
            double delayR = base_samps + stereo_samps + sin(2.0 * M_PI * lfoR) * depth_samps;

            // write current filtered samples to delay buffers
            x->dlyL[wr] = wl;
            x->dlyR[wr] = wrs;

            // read
            double tapL = tap_delay(x->dlyL, size, wr, delayL);
            double tapR = tap_delay(x->dlyR, size, wr, delayR);

            // mix (simple chorus)
            double mix = 0.35 + 0.65 * water; // 0.35..1.0
            dl = (1.0 - mix) * wl + mix * tapL;
            dr = (1.0 - mix) * wrs + mix * tapR;

            // advance LFOs
            lfoL += lfo_inc; if (lfoL >= 1.0) lfoL -= 1.0;
            lfoR += lfo_inc * 0.97; if (lfoR >= 1.0) lfoR -= 1.0; // slight rate offset for extra swirl

            // advance write pointer
            wr++; if (wr >= size) wr = 0;
        }
        else {
            // still write to keep buffers warm
            x->dlyL[wr] = wl; x->dlyR[wr] = wrs; wr++; if (wr >= size) wr = 0;
        }

        // Add thickness
        double thick = x->thickness * 0.7;
        wl = wl * (1.0 + thick * (1.0 - env));
        wrs = wrs * (1.0 + thick * (1.0 - env));

        // Oversample before soft clipper
        if (soft > 0.0) {
            // 2x oversampling
            double wl2[2] = {wl, x->osL.z1};
            double wr2[2] = {wrs, x->osR.z1};

            for (int j = 0; j < 2; j++) {
                wl2[j] = biquad_process(&x->osL.up[j], wl2[j]);
                wr2[j] = biquad_process(&x->osR.up[j], wr2[j]);

                wl2[j] = softclip(wl2[j], soft);
                wr2[j] = softclip(wr2[j], soft);

                wl2[j] = biquad_process(&x->osL.down[j], wl2[j]);
                wr2[j] = biquad_process(&x->osR.down[j], wr2[j]);
            }

            wl = wl2[0];
            wrs = wr2[0];
            x->osL.z1 = wl2[1];
            x->osR.z1 = wr2[1];
        }

        // Dynamic envelope following
        double in_abs = fabs(l + r) * 0.5;
        x->env_follow = in_abs + x->env_smooth * (x->env_follow - in_abs);
        double dyn_mod = 1.0 + x->dynamics * (x->env_follow - 0.5);

        // Enhance modulation with dynamics
        mod *= (1.0 + 0.2 * dyn_mod);

        // Process through resonators
        double res_amtL = x->bodyres * (0.7 + 0.3 * env);
        double res_amtR = res_amtL * 0.97; // slight stereo variation

        wl = wl + resonator_process(&x->resL, wl) * res_amtL;
        wrs = wrs + resonator_process(&x->resR, wrs) * res_amtR;

        // Add warmth with subtle saturation
        if (x->warmth > 0.0) {
            double warm = x->warmth * (0.5 + 0.5 * env);
            wl = wl + warm * tanh(wl * 1.5) * 0.33;
            wrs = wrs + warm * tanh(wrs * 1.5) * 0.33;
        }

        // Apply research-based filtering
        wl = biquad_process(&x->mtf_filter[0], wl);
        wl = biquad_process(&x->voice_enhance[0], wl);
        wrs = biquad_process(&x->mtf_filter[1], wrs);
        wrs = biquad_process(&x->voice_enhance[1], wrs);

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
        double ol = dl * wet + l * (1.0 - wet);
        double orr = dr * wet + r * (1.0 - wet);

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
    }
}

void wombifier_set_q(t_wombifier *x, void *attr, long ac, t_atom *av) {
    if (ac && av) {
        double v = atom_getfloat(av);
        x->q = clampd(v, 0.3, 8.0);
        x->coeffs_dirty = 1; 
    }
}