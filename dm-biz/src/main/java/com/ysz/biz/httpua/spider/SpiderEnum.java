package com.ysz.biz.httpua.spider;

import com.google.common.collect.ImmutableSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import org.apache.commons.lang.StringUtils;

public enum SpiderEnum {

  /**
   * 百度引擎
   */
  baidu("baiduspider"),
  /**
   * google 引擎
   */
  google("googlebot"),
  /**
   * 搜索搜索
   */
  sogou("sogou web spider/4.0",
      "sogou inst spider/4.0",
      "sogou news spider/4.0",
      "sogou pic spider/3.0",
      "sogou spider",
      "sogou video spider/3.0"),

  /**
   * 字节搜索
   */
  toutiao("bytespider"),

  /**
   * 360 搜索
   */
  so_360("360spider"),


  bing("bingbot/2.0", "bingpreview", "bingweb"),

  apple("applebot"),

  yisou("yisouspider"),

  soso("sosospider", "sosoimagespider");


  private final ImmutableSet<String> keywords;

  SpiderEnum(String... keywordArray) {
    this.keywords = ImmutableSet.copyOf(keywordArray);
  }

  public static Set<SpiderEnum> detectAll(String httpAgent) {
    if (StringUtils.isBlank(httpAgent)) {
      return Collections.emptySet();
    }

    final String lowerCase = StringUtils.lowerCase(httpAgent);
    Set<SpiderEnum> result = new HashSet<>();
    for (SpiderEnum value : SpiderEnum.values()) {
      final ImmutableSet<String> keywords = value.keywords;
      for (String keyword : keywords) {
        if (lowerCase.contains(keyword)) {
          result.add(value);
          break;
        }
      }


    }

    return result;
  }


}
