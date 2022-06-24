package com.ysz.biz.httpua;

import nl.basjes.parse.useragent.UserAgent;
import nl.basjes.parse.useragent.UserAgentAnalyzer;

public class HttpUa {


  public static void main(String[] args) throws Exception {
    UserAgentAnalyzer uaa = UserAgentAnalyzer
        .newBuilder()
        .hideMatcherLoadStats()
        .withCache(10000)
        .build();

    UserAgent agent = uaa.parse(
        "Baiduspider+(+http://www.baidu.com/search/spider.htm);googlebot|baiduspider|baidu|spider|sogou|bingbot|bot|yahoo|soso|sosospider|360spider|youdaobot|jikeSpider;)");

    System.out.println(agent.hasSyntaxError());
    for (String fieldName : agent.getAvailableFieldNamesSorted()) {
      System.out.println(fieldName + " = " + agent.getValue(fieldName));

    }
  }
}
