package com.ysz.dm.fast.basic.io.append;

import java.io.File;
import org.apache.commons.io.FileUtils;

public class IoAppend_Dm_001 {

  public static void main(String[] args) throws Exception {
    File file = new File("/Users/carl/tmp/useless/1.txt");
    FileUtils.write(file, "this is the first", true);
  }

}
