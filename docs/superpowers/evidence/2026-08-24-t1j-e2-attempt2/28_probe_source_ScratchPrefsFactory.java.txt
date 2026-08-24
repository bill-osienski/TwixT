package e2probe;

import java.util.prefs.Preferences;
import java.util.prefs.PreferencesFactory;

/** Selected via -Djava.util.prefs.PreferencesFactory before any T1j class loads. */
public class ScratchPrefsFactory implements PreferencesFactory {
    private static final ScratchPrefs USER = new ScratchPrefs(null, "");
    private static final ScratchPrefs SYSTEM = new ScratchPrefs(null, "");
    public Preferences userRoot() { return USER; }
    public Preferences systemRoot() { return SYSTEM; }
}
