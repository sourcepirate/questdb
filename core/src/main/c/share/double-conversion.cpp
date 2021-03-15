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

#include "double-conversion.h"
#include "double-conversion/double-conversion.h"

using namespace double_conversion;

JNIEXPORT jint JNICALL Java_io_questdb_std_DoubleConversion_append
        (JNIEnv *env, jclass clazz, jlong ptr, jdouble value) {
    char *p = reinterpret_cast<char *>(ptr);
    bool sign;
    int length;
    int point;
    DoubleToStringConverter::DoubleToAscii(value, DoubleToStringConverter::DtoaMode::SHORTEST, 19, p, 4096,
                                           &sign, &length, &point);
    char *pointInBuffer = p + point;
    memmove(pointInBuffer + 1, pointInBuffer, length - point);
    *pointInBuffer = '.';
    return length;
}

static StringToDoubleConverter converter(StringToDoubleConverter::NO_FLAGS, 0.0, 0.0, "Infinity", "NaN");

JNIEXPORT jdouble JNICALL Java_io_questdb_std_DoubleConversion_parse
        (JNIEnv *env, jclass clazz, jlong charArrayPtr, jint length) {
    int processed;
    return converter.StringToDouble(reinterpret_cast<const char *>(charArrayPtr), length, &processed);
}
