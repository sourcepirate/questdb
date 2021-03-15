/*******************************************************************************
 *     ___                  _   ____  ____
 *    / _ \ _   _  ___  ___| |_|  _ \| __ )
 *   | | | | | | |/ _ \/ __| __| | | |  _ \
 *   | |_| | |_| |  __/\__ \ |_| |_| | |_) |
 *    \__\_\\__,_|\___||___/\__|____/|____/
 *
 *  Copyright (c) 2014-2019 Appsicle
 *  Copyright (c) 2019-2020 QuestDB
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 ******************************************************************************/

package org.questdb;

import io.questdb.std.*;
import io.questdb.std.str.DirectCharSink;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.concurrent.TimeUnit;

@State(Scope.Thread)
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
public class DoubleConversionBenchmark {
    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(DoubleConversionBenchmark.class.getSimpleName())
                .warmupIterations(5)
                .measurementIterations(5)
                .forks(1)
                .build();

        new Runner(opt).run();
    }

    private static final int nativeMemorySize = 4096;
    private static final String asString = "584987.587898";
    private static final int asStringLength = 13;
    private static final double asDouble = 584987.587898D;

    static {
        assert asString.length() == asStringLength;
    }

    @State(Scope.Benchmark)
    public static class MyState {
        public long to = Unsafe.malloc(nativeMemorySize);
        public DirectCharSink sink = new DirectCharSink(nativeMemorySize); // When running this benchmark, comment out every `this.lo +=` line.
        public long from = Chars.toMemory(asString);
    }

    @Benchmark
    public int testDoubleToStringC(MyState state) {
        return DoubleConversion.append(state.to, asDouble);
    }

    @Benchmark
    public void testDoubleToStringJava(MyState state) {
        Numbers.append(state.sink, asDouble);
    }

    @Benchmark
    public double testStringToDoubleC(MyState state) {
        return DoubleConversion.parse(state.from, asStringLength);
    }

    @Benchmark
    public double testStringToDoubleJava() throws NumericException {
        return Numbers.parseDouble(asString);
    }
}
