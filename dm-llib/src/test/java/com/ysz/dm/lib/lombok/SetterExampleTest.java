package com.ysz.dm.lib.lombok;

import org.junit.Test;

/**
 * @author carl
 * @create 2022-10-26 4:56 PM
 **/
public class SetterExampleTest {

  @Test
  public void testCreate() {
    SetterExample setterExample = SetterExample.of(10L);
    System.out.println(setterExample.userId());
  }

}