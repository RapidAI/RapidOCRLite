package ai.rapidocr.lite;

import java.util.Arrays;

/**
 * One recognized text line: the text, a confidence score and the geometry.
 * Instances are immutable and safe to share.
 */
public final class TextLine {

    private final String text;
    private final float confidence;
    /** Four corners in order TL, TR, BR, BL; each is {x, y}. */
    private final float[][] box;
    private final int bboxX;
    private final int bboxY;
    private final int bboxW;
    private final int bboxH;
    /** Orientation class index (0 = "0", 1 = "180"), -1 when not classified. */
    private final int clsLabel;
    private final float clsScore;
    private static final Word[] NO_WORDS = new Word[0];
    private final Word[] words;
    private final Word[] charBoxes;
    private final int recSteps;
    private final int cropWidth;
    private final int cropHeight;
    private final double recMaxWhRatio;
    private final int[] charCols;
    private final float[] charProbs;

    /**
     * Called by the JNI layer; application code normally only reads results.
     *
     * @param text       recognized text (never null)
     * @param confidence 0..1 score
     * @param box        4x2 corners, TL, TR, BR, BL
     */
    public TextLine(String text, float confidence, float[][] box,
                    int bboxX, int bboxY, int bboxW, int bboxH) {
        this(text, confidence, box, bboxX, bboxY, bboxW, bboxH, -1, 0f);
    }

    /**
     * @param clsLabel text-line orientation class (0 = "0", 1 = "180"), -1 when
     *                 the classifier did not run; the line was rotated 180 degrees
     *                 before recognition when 1 scored above cls_thresh
     * @param clsScore probability of {@code clsLabel}
     */
    public TextLine(String text, float confidence, float[][] box,
                    int bboxX, int bboxY, int bboxW, int bboxH, int clsLabel, float clsScore) {
        this(text, confidence, box, bboxX, bboxY, bboxW, bboxH, clsLabel, clsScore,
                null, null, 0, 0, 0, 0.0, null, null);
    }

    /**
     * Full form used by the JNI layer: also the word boxes, the character
     * boxes and the per-character recognizer data they are computed from.
     */
    public TextLine(String text, float confidence, float[][] box,
                    int bboxX, int bboxY, int bboxW, int bboxH, int clsLabel, float clsScore,
                    Word[] words, Word[] charBoxes, int recSteps, int cropWidth, int cropHeight,
                    double recMaxWhRatio, int[] charCols, float[] charProbs) {
        this.words = words == null ? NO_WORDS : words;
        this.charBoxes = charBoxes == null ? NO_WORDS : charBoxes;
        this.recSteps = recSteps;
        this.cropWidth = cropWidth;
        this.cropHeight = cropHeight;
        this.recMaxWhRatio = recMaxWhRatio;
        this.charCols = charCols == null ? new int[0] : charCols;
        this.charProbs = charProbs == null ? new float[0] : charProbs;
        this.clsLabel = clsLabel;
        this.clsScore = clsScore;
        this.text = text == null ? "" : text;
        this.confidence = confidence;
        this.box = box;
        this.bboxX = bboxX;
        this.bboxY = bboxY;
        this.bboxW = bboxW;
        this.bboxH = bboxH;
    }

    public String text() {
        return text;
    }

    public float confidence() {
        return confidence;
    }

    /** Defensive copy of the four corners TL, TR, BR, BL. */
    public float[][] box() {
        float[][] copy = new float[box.length][];
        for (int i = 0; i < box.length; i++) {
            copy[i] = box[i] == null ? null : box[i].clone();
        }
        return copy;
    }

    public int bboxX() {
        return bboxX;
    }

    public int bboxY() {
        return bboxY;
    }

    public int bboxW() {
        return bboxW;
    }

    public int bboxH() {
        return bboxH;
    }

    /** Orientation class index (0 = "0", 1 = "180"); -1 when not classified. */
    public int clsLabel() {
        return clsLabel;
    }

    /** Probability of {@link #clsLabel()}. */
    public float clsScore() {
        return clsScore;
    }

    /**
     * Word boxes (rapidocr main {@code return_word_box=True}): English and
     * number runs as words, CJK as single characters. Empty for blank lines.
     */
    public java.util.List<Word> words() {
        return java.util.Collections.unmodifiableList(Arrays.asList(words));
    }

    /** One box per character ({@code return_single_char_box=True}). */
    public java.util.List<Word> charBoxes() {
        return java.util.Collections.unmodifiableList(Arrays.asList(charBoxes));
    }

    /** {@link #charBoxes()} when {@code singleChar}, else {@link #words()}. */
    public java.util.List<Word> words(boolean singleChar) {
        return singleChar ? charBoxes() : words();
    }

    /** CTC output length of the line's recognizer batch; 0 when not recorded. */
    public int recSteps() {
        return recSteps;
    }

    /** Width of the line crop the recognizer read (the word-box geometry). */
    public int cropWidth() {
        return cropWidth;
    }

    public int cropHeight() {
        return cropHeight;
    }

    /** The batch's max width/height ratio (main's max_wh_ratio). */
    public double recMaxWhRatio() {
        return recMaxWhRatio;
    }

    /** CTC time step of each decoded character. */
    public int[] charCols() {
        return charCols.clone();
    }

    /** CTC probability of each decoded character. */
    public float[] charProbs() {
        return charProbs.clone();
    }

    @Override
    public String toString() {
        return String.format("TextLine[%.3f] \"%s\" box=%s", confidence, text,
                Arrays.toString(box));
    }
}
