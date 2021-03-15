package io.questdb.std;

import org.junit.Assert;
import org.junit.Test;

public class DoubleConversionTest {
    private static final double asDouble = 587.0658D;
    private static final String asString = Double.toString(asDouble);
    private static final int asStringLength = asString.length();

    @Test
    public void append() {
        long ptr = Unsafe.malloc(4096L);
        DoubleConversion.append(ptr, asDouble);
        Assert.assertEquals(asString, Chars.asciiStrRead(ptr, asStringLength));
    }

    @Test
    public void parse() {
        long ptr = Chars.toMemory(asString);
        double value = DoubleConversion.parse(ptr, asStringLength);
        Assert.assertEquals(asDouble, value, 0D);
    }
}
