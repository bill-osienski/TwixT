package e2probe;

import java.util.Map;
import java.util.TreeMap;
import java.util.prefs.AbstractPreferences;

/** In-memory Preferences node. Never touches the host preference store. */
public class ScratchPrefs extends AbstractPreferences {
    private final Map<String, String> vals = new TreeMap<String, String>();
    private final Map<String, ScratchPrefs> kids = new TreeMap<String, ScratchPrefs>();

    public ScratchPrefs(ScratchPrefs parent, String name) { super(parent, name); }

    protected void putSpi(String k, String v) { vals.put(k, v); }
    protected String getSpi(String k) { return vals.get(k); }
    protected void removeSpi(String k) { vals.remove(k); }
    protected void removeNodeSpi() { vals.clear(); kids.clear(); }
    protected String[] keysSpi() { return vals.keySet().toArray(new String[0]); }
    protected String[] childrenNamesSpi() { return kids.keySet().toArray(new String[0]); }
    protected AbstractPreferences childSpi(String name) {
        ScratchPrefs c = kids.get(name);
        if (c == null) { c = new ScratchPrefs(this, name); kids.put(name, c); }
        return c;
    }
    protected void syncSpi() { }
    protected void flushSpi() { }
}
