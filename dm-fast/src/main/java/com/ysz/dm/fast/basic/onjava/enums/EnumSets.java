package com.ysz.dm.fast.basic.onjava.enums;

import java.util.EnumSet;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/26
 **/
public class EnumSets {

  public static void main(String[] args) {
    /*Empty*/
    EnumSet<AlarmPoints> points = EnumSet.noneOf(AlarmPoints.class);

    points.add(AlarmPoints.BATHROOM);
    System.out.println(points);

    points.addAll(EnumSet.of(AlarmPoints.STAIR1, AlarmPoints.STAIR2, AlarmPoints.KITCHEN));
    System.out.println(points);
  }
}
