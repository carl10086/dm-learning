package com.ysz.dm.fast.basic;

import com.google.common.collect.Lists;
import java.math.BigDecimal;
import java.text.ParseException;
import java.util.Date;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;
import org.apache.commons.lang3.time.FastDateFormat;

public class Npl_001 {

  private static long start;
  private static final long UNIT = 60L * 3600L * 1000L;

  {
    try {
      start = FastDateFormat.getInstance("yyyy-MM-dd HH:mm:ss").parse("2014-06-15 10:00:00")
          .getTime();
    } catch (ParseException e) {
    }
  }


  private static float score(int vote, Date createAt) {
    long t = createAt.getTime() - start;
    float rightPart = new BigDecimal(t).divide(new BigDecimal(UNIT), 4, BigDecimal.ROUND_CEILING)
        .floatValue();
    if (vote == 0) {
      return rightPart;
    }
    float leftPart = (float) (Math.log(vote) / Math.log(2));
    return leftPart + rightPart;
  }

  public static void main(String[] args) throws Exception {
    /*1 在*/
    List<String> data = Lists.newArrayList("1", "2", "3", "4");

    /*4个 -> yarn 调度器 ... yarn -> master -> job 2 _> 2*/

    data.stream().map(x -> x + "a").filter(Objects::nonNull).sorted().collect(Collectors.toList());

  }


}
