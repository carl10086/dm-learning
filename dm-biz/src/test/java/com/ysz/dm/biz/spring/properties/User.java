package com.ysz.dm.biz.spring.properties;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * @author carl
 * @create 2022-09-07 1:06 PM
 **/
@ConfigurationProperties(prefix = "user")
//@ConfigurationPropertiesScan
public class User {

  private String name;

  private String fuck;


  public String getName() {
    return name;
  }

  public User setName(String name) {
    this.name = name;
    return this;
  }

  public String getFuck() {
    return fuck;
  }

  public User setFuck(String fuck) {
    this.fuck = fuck;
    return this;
  }

  @Override
  public String toString() {
    return "User{" +
        "name='" + name + '\'' +
        ", fuck='" + fuck + '\'' +
        '}';
  }
}
