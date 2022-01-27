package com.ysz.biz.score;

import com.google.common.base.Splitter;
import java.io.File;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.nio.charset.Charset;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.stream.Collectors;
import lombok.Data;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang3.time.FastDateFormat;

public class ScoreTmp {

  public static final long START = start();

  private static long start() {
    try {
      return FastDateFormat.getInstance("yyyy-MM-dd").parse("2011-01-01").getTime();
    } catch (Exception e) {
      return 0L;
    }
  }

  public void execute() throws Exception {
    String filePath = "/Users/carl/soft/tmp/fuck/c.txt";
    final List<String> strings = FileUtils.readLines(new File(filePath), Charset.defaultCharset());
    List<Item> items = new ArrayList<>(strings.size() - 1);
    for (int i = 0; i < strings.size(); i++) {
      items.add(new Item(strings.get(i)));
    }
    final long max = items.stream().mapToLong(x -> x.getCreateAt().longValue()).max().getAsLong();
    long now = max + Duration.ofHours(1L).toMillis();
    Collections.sort(items, (o1, o2) -> o2.createAt.compareTo(o1.createAt));

    /*时间衰弱因子*/

    double[] timeWeakFactorArray = new double[]{1.02};
    for (int i = 0; i < timeWeakFactorArray.length; i++) {
      double v = timeWeakFactorArray[i];
      for (Item item : items) {
//             * x 是分数因子. x 越小, cnt 占比越重
//            * y 是时间因子, y 越大, 时间占比越轻, y 越小时间越重要 .
        item.score(v, Duration.ofHours(1L).toMillis(), now);
      }

      final List<String> collect = items.stream().sorted((o1, o2) -> o2.getScore().compareTo(o1.getScore())).limit(40)
          .map(Item::toString).collect(
              Collectors.toList());

      FileUtils.writeLines(new File("/Users/carl/soft/tmp/fuck/score/1"), collect);

    }

  }


  private long dats(Item x, long now) {
    final long days = (now - x.getCreateAt()) / Duration.ofDays(1L).toMillis();
    if (days <= 1) {
      return 1;
    }
    if (days <= 7) {
      return 7;
    }
    if (days <= 30) {
      return 30;
    }
    if (days <= 60) {
      return 60;
    }
    if (days <= 30 * 6) {
      return 30 * 6;
    }
    if (days <= 30 * 12) {
      return 30 * 12;
    }
    return 30 * 24;
  }


  public static double log(double x, double n) {
    return Math.log(x) / Math.log(n);
  }

  public static void main(String[] args) throws Exception {
    new ScoreTmp().execute();
  }

  @Data
  private static class Item {

    private static final Splitter splitter = Splitter.on("\t");

    private Long atlasId;

    private Integer comment;

    private Integer like;

    private Long cnt;

    private Integer fav;

    private Long createAt;

    private Double score;

    private double timePart = 0.0;
    private double cntPart = 0.0;

    private Long duration;

    public Item(String s) {
      final List<String> strings = splitter.splitToList(s);
      this.atlasId = Long.valueOf(strings.get(0));
      this.comment = Long.valueOf(strings.get(1)).intValue();
      this.like = Long.valueOf(strings.get(2)).intValue();
      this.fav = Long.valueOf(strings.get(3)).intValue();
      this.createAt = Long.valueOf(strings.get(4));
    }


    /**
     * x 是分数因子. x 越小, cnt 占比越重
     * y 是时间因子, y 越大, 时间占比越轻, y 越小时间越重要 .
     * @param x
     * @param y
     */
    public void score(double x, long y, long end) {
      this.duration = (end - createAt);
      Long cnt = (long) (comment * 2 + like + fav);
      this.cnt = cnt;
      long minDuration = Duration.ofHours(12L).toMillis();
      if (this.duration <= minDuration) {
        this.duration = minDuration;
      }
      int max = 1300;
      this.timePart = max - log(duration, x);

      this.cntPart = cnt;

      this.score = this.timePart + this.cntPart;
    }


    @Override

    public String toString() {
      return String
          .format("Atlas{atlasId:%s,创建时间:%s小时之前, 互动数:%s, 分数:%s, 时间占比:%s, 互动占比:%s}", atlasId, duration / 1000L / 3600L,
              cnt, score, timePart, cntPart);
//      return String
//          .format("Atlas{创建时间:%s小时之前, 互动数:%s, 分数:%s}", duration / 1000L / 3600L, comment * 2 + like + fav, score);
    }
  }
}
