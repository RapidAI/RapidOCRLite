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

    /**
     * Called by the JNI layer; application code normally only reads results.
     *
     * @param text       recognized text (never null)
     * @param confidence 0..1 score
     * @param box        4x2 corners, TL, TR, BR, BL
     */
    public TextLine(String text, float confidence, float[][] box,
                    int bboxX, int bboxY, int bboxW, int bboxH) {
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

    @Override
    public String toString() {
        return String.format("TextLine[%.3f] \"%s\" box=%s", confidence, text,
                Arrays.toString(box));
    }
}
