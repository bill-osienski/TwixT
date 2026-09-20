# The committed `.class` files are first-party probes

Several evidence directories carry a small `*_classes/` tree of compiled Java
classes, 52 files and about 341 KB in total:

    e2probe/ScratchPrefs.class
    e2probe/ScratchPrefsFactory.class
    net/schwagereit/t1j/E3bDump.class
    net/schwagereit/t1j/E4Preflight.class

**All four are OURS.** `E3bDump` and `E4Preflight` are probe classes written in
this repository; they are declared in the `net.schwagereit.t1j` package only
because Java requires a class to sit in the package whose package-private
members it reads. `ScratchPrefs` and `ScratchPrefsFactory` are our own
preference-surface stubs. Their sources live under `scripts/` and `probes/` in
this repository and the classes are reproducible from them.

**No T1j jar, class or source file is included anywhere in this repository.**
The verified T1j toolchain is deliberately held OUTSIDE the repository — see
the E1 artifact-integrity record — and is referenced by hash, never vendored.
A `.class` sitting under a `net/schwagereit/t1j/` path is our probe compiled
into that namespace, not redistributed third-party code.

They are committed rather than regenerated because each one is the *exact*
binary a recorded run executed: the run records pin its sha256, and a probe
recompiled later under a different JDK would not reproduce that hash. They are
evidence, not build output.
