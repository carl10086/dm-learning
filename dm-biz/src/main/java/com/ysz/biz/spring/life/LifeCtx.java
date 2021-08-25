package com.ysz.biz.spring.life;

import com.alibaba.nacos.api.annotation.NacosProperties;
import com.alibaba.nacos.spring.context.annotation.config.NacosPropertySource;
import com.alibaba.nacos.spring.context.annotation.config.NacosPropertySources;
import com.ysz.biz.spring.life.anno.EnableMyBeanAnno;
import com.ysz.biz.spring.life.anno.EnableMyNacosAnno;
import lombok.extern.slf4j.Slf4j;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
@ComponentScan("com.ysz.biz.spring.life")
@PropertySource("classpath:tst.properties")
//@EnableMyNacosAnno(globalProperties =
//@NacosProperties(serverAddr = "10.1.1.63:3000",
//    enableRemoteSyncConfig = "true" /*监听器首先添加的时候拉取远端配置, 默认值 false*/,
//    maxRetry = "5" /*长轮询的重试次数、默认 3*/,
//    configRetryTime = "4000",
//    configLongPollTimeout = "26000")
//)
//@NacosPropertySources(
//    {
//        @NacosPropertySource(dataId = "dt-user", autoRefreshed = true, groupId = "SOC"),
//        @NacosPropertySource(dataId = "dt-infra", autoRefreshed = true, groupId = "COMMON")
//    }
//)
@EnableMyBeanAnno
@Slf4j
public class LifeCtx {

  public LifeCtx() {
    log.info("LifeCtx constructed");
  }
}
