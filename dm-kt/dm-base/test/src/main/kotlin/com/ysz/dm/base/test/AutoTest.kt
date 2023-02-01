package com.ysz.dm.base.test

import org.junit.jupiter.api.Assertions

/**
 * @author carl
 * @since 2023-02-01 2:39 PM
 **/
infix fun <T> T.eq(rval: T) {
    Assertions.assertEquals(this, rval)
}

infix fun <T> Array<T>.eq(rval: Array<T>) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun IntArray.eq(rval: IntArray) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun ByteArray.eq(rval: ByteArray) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun FloatArray.eq(rval: FloatArray) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun LongArray.eq(rval: LongArray) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun DoubleArray.eq(rval: DoubleArray) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun ShortArray.eq(rval: ShortArray) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun BooleanArray.eq(rval: BooleanArray) {
    Assertions.assertArrayEquals(this, rval)
}

infix fun CharArray.eq(rval: CharArray) {
    Assertions.assertArrayEquals(this, rval)
}