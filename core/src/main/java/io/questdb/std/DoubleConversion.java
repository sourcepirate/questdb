package io.questdb.std;

public class DoubleConversion {
    static {
        Os.init();
    }

    public static native int append(long ptr, double value);

    public static native double parse(long charArrayPtr, int length);
}
