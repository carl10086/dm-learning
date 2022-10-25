package com.ysz.dm.lib.lombok;

import java.util.Set;
import lombok.Builder;
import lombok.Getter;
import lombok.Singular;
import lombok.ToString;

/**
 * @author carl
 * @create 2022-10-25 6:56 PM
 **/
@Builder
@ToString
@Getter
public class BuilderExample {

  @Builder.Default
  private long created = System.currentTimeMillis();
  private String name;
  private int age;
  @Singular
  private Set<String> occupations;
}
