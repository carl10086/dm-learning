package com.ysz.dm.base.hbase

import org.apache.hadoop.hbase.util.Bytes

/**
 * @author carl
 * @since 2022-12-29 8:40 PM
 **/
class HBaseTools {

    companion object {

        private fun fillZeroPrefix(uid: String, fixedLength: Int): String {
            require(uid.length <= fixedLength) { "长度超过了11位:" + uid.length }
            if (uid.length == fixedLength) {
                return uid
            }
            val prefixLen: Int = fixedLength - uid.length
            val sb = StringBuilder()
            for (i in 0 until prefixLen) {
                sb.append("0")
            }
            return sb.toString() + uid
        }

        fun normalize(id: String, fixedLength: Int = 11) = fillZeroPrefix(id, fixedLength).reversed()


        /**
         * convert a double array to bytes
         */
        fun doubleArrayToBytes(doubleArray: DoubleArray): ByteArray {
            if (doubleArray.isEmpty()) return ByteArray(0)

            val result = ByteArray(8 * doubleArray.size)

            var index = 0

            for (d in doubleArray) {
                Bytes.toBytes(d).forEach {
                    result[index++] = it
                }
            }

            return result
        }

        fun toDoubleArray(byteArray: ByteArray): DoubleArray {

            if (byteArray.isEmpty()) return DoubleArray(0)

            val size = byteArray.size

            check(size % 8 == 0) { "one double need 8 bytes, so byte array size must % 8 == 0" }


            val len = size / 8
            val result = DoubleArray(len)

            for (i in 0 until len) {
                // 0 : 0->7
                // 1 : 8->15
                result[i] = Bytes.toDouble(byteArray, i * 8)
            }

            return result
        }

    }
}