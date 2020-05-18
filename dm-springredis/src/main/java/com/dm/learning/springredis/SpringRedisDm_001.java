package com.dm.learning.springredis;

import org.springframework.context.annotation.AnnotationConfigApplicationContext;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.EnableAspectJAutoProxy;
import org.springframework.data.redis.core.StringRedisTemplate;

@Configuration
@EnableAspectJAutoProxy
public class SpringRedisDm_001 {

  @Bean
  public StringRedisTemplate stringRedisTemplate() {
    StringRedisTemplate stringRedisTemplate = JedisUtils.redisCacheTemplate(
        "127.0.0.1",
        6379
    );
    return stringRedisTemplate;
  }

  public static void main(String[] args) {
    AnnotationConfigApplicationContext ctx = new AnnotationConfigApplicationContext(
        SpringRedisDm_001.class
    );
    StringRedisTemplate bean = ctx.getBean(StringRedisTemplate.class);
    Boolean blog_comment = bean.hasKey("0:0.1163141788");
    System.out.println(blog_comment);
  }

}
