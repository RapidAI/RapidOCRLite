package ai.rapidocr.lite;

/**
 * One word or character box of a {@link TextLine}, rapidocr main's
 * {@code word_results} item {@code (text, confidence, box)}. English and
 * number runs are words, CJK text is one entry per character; see
 * {@link TextLine#words()} and {@link TextLine#charBoxes()}. Immutable.
 */
public final class Word {

    private final String text;
    private final double confidence;
    private final int[][] box;

    /** Called by the JNI layer. */
    public Word(String text, double confidence, int[][] box) {
        this.text = text == null ? "" : text;
        this.confidence = confidence;
        this.box = box;
    }

    public String text() {
        return text;
    }

    /** main's per-character score, {@code round(prob, 5)}, paired the way main pairs it. */
    public double confidence() {
        return confidence;
    }

    /** Defensive copy of the four integer corners {x, y} (main's order_points order). */
    public int[][] box() {
        int[][] copy = new int[box.length][];
        for (int i = 0; i < box.length; i++) {
            copy[i] = box[i].clone();
        }
        return copy;
    }

    @Override
    public String toString() {
        StringBuilder b = new StringBuilder();
        for (int[] p : box) {
            b.append('(').append(p[0]).append(',').append(p[1]).append(')');
        }
        return "Word[" + confidence + "] \"" + text + "\" " + b;
    }
}
