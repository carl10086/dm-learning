package com.ysz.biz.spring.life;

import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.core.env.Environment;

public class LifeApp {

  public static void main(String[] args) {
    AnnotationConfigApplicationContext ctx = new AnnotationConfigApplicationContext(LifeCtx.class);
    ctx.getBean(LifeBean.class).sayHi("carl");
    Environment bean = ctx.getBean(Environment.class);
    System.out.println(bean.getProperty("fire"));
    ctx.close();
  }
}
