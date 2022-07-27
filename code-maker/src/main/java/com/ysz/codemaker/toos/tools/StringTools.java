package com.ysz.codemaker.toos.tools;

import com.google.common.base.CaseFormat;
import com.google.common.base.Joiner;
import com.google.common.base.Strings;
import com.ysz.codemaker.mybatis.core.DefaultMappingStrategy;

public class StringTools {

  public static final Joiner JOINER = Joiner.on(",").skipNulls();


  public static String formatConvert(
      DefaultMappingStrategy mappingStrategy,
      String origin
  ) {
    if (Strings.isNullOrEmpty(origin)) {
      return origin;
    }

    switch (mappingStrategy) {
      case LOWER_UNDERSCORE_2_LOWER_CAMEL:
        return CaseFormat.LOWER_UNDERSCORE.to(CaseFormat.LOWER_CAMEL, origin);
      default:
        return origin;
    }

  }

}
