package com.ysz.dm.netty.dm.http.util.unit;

public class ByteSizeValue {

  private final long size;
  private final ByteSizeUnit unit;

  public ByteSizeValue(long bytes) {
    this(bytes, ByteSizeUnit.BYTES);
  }

  public ByteSizeValue(long size, ByteSizeUnit unit) {
    if (size < -1 || (size == -1 && unit != ByteSizeUnit.BYTES)) {
      throw new IllegalArgumentException(
          "Values less than -1 bytes are not supported: " + size + unit.getSuffix());
    }
    if (size > Long.MAX_VALUE / unit.toBytes(1)) {
      throw new IllegalArgumentException(
          "Values greater than " + Long.MAX_VALUE + " bytes are not supported: " + size + unit
              .getSuffix());
    }
    this.size = size;
    this.unit = unit;
  }


  public long getBytes() {
    return unit.toBytes(size);
  }

  public long getKb() {
    return unit.toKB(size);
  }

  public long getMb() {
    return unit.toMB(size);
  }

  public long getGb() {
    return unit.toGB(size);
  }

  public long getTb() {
    return unit.toTB(size);
  }

  public long getPb() {
    return unit.toPB(size);
  }

  public double getKbFrac() {
    return ((double) getBytes()) / ByteSizeUnit.C1;
  }

  public double getMbFrac() {
    return ((double) getBytes()) / ByteSizeUnit.C2;
  }

  public double getGbFrac() {
    return ((double) getBytes()) / ByteSizeUnit.C3;
  }

  public double getTbFrac() {
    return ((double) getBytes()) / ByteSizeUnit.C4;
  }

  public double getPbFrac() {
    return ((double) getBytes()) / ByteSizeUnit.C5;
  }


  @Override
  public String toString() {
    long bytes = getBytes();
    double value = bytes;
    String suffix = ByteSizeUnit.BYTES.getSuffix();
    if (bytes >= ByteSizeUnit.C5) {
      value = getPbFrac();
      suffix = ByteSizeUnit.PB.getSuffix();
    } else if (bytes >= ByteSizeUnit.C4) {
      value = getTbFrac();
      suffix = ByteSizeUnit.TB.getSuffix();
    } else if (bytes >= ByteSizeUnit.C3) {
      value = getGbFrac();
      suffix = ByteSizeUnit.GB.getSuffix();
    } else if (bytes >= ByteSizeUnit.C2) {
      value = getMbFrac();
      suffix = ByteSizeUnit.MB.getSuffix();
    } else if (bytes >= ByteSizeUnit.C1) {
      value = getKbFrac();
      suffix = ByteSizeUnit.KB.getSuffix();
    }
    return format1Decimals(value, suffix);
  }

  private static String format1Decimals(double value, String suffix) {
    String p = String.valueOf(value);
    int ix = p.indexOf('.') + 1;
    int ex = p.indexOf('E');
    char fraction = p.charAt(ix);
    if (fraction == '0') {
      if (ex != -1) {
        return p.substring(0, ix - 1) + p.substring(ex) + suffix;
      } else {
        return p.substring(0, ix - 1) + suffix;
      }
    } else {
      if (ex != -1) {
        return p.substring(0, ix) + fraction + p.substring(ex) + suffix;
      } else {
        return p.substring(0, ix) + fraction + suffix;
      }
    }
  }
}
