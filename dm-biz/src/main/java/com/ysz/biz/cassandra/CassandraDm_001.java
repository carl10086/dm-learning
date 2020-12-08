package com.ysz.biz.cassandra;

import com.google.common.hash.Hashing;
import java.nio.ByteBuffer;
import java.nio.charset.Charset;
import java.util.Arrays;
import org.apache.commons.codec.digest.MurmurHash3;
import org.apache.commons.lang3.SerializationUtils;
import org.apache.hadoop.fs.ByteBufferUtil;
import org.apache.hadoop.hbase.util.ByteBufferUtils;

public class CassandraDm_001 {


  public static void putLong(byte[] b, int off, long val) {
    b[off + 7] = (byte) (val);
    b[off + 6] = (byte) (val >>> 8);
    b[off + 5] = (byte) (val >>> 16);
    b[off + 4] = (byte) (val >>> 24);
    b[off + 3] = (byte) (val >>> 32);
    b[off + 2] = (byte) (val >>> 40);
    b[off + 1] = (byte) (val >>> 48);
    b[off] = (byte) (val >>> 56);
  }

  private static long[] getHash(long n) {
    ByteBuffer key = ByteBuffer.allocate(8).putLong(0, n);
    long[] hash = new long[2];
    MurmurHash.hash3_x64_128(key, key.position(), key.remaining(), 0, hash);
    return hash;
  }


  public static void main(String[] args) {
//    System.err.println(Arrays.toString(MurmurHash3.hash128x64()));
//    System.err.println(Arrays.toString(MurmurHash3.hash128x64(Bytes.toBytes(2L))));
//    System.err.println(Arrays.toString(MurmurHash3.hash128x64(Bytes.toBytes(3L))));
//    System.err.println(Arrays.toString(MurmurHash3.hash128x64(Bytes.toBytes(507862512L))));

    System.err.println(Arrays.toString(getHash(28450830L)));
  }

}
