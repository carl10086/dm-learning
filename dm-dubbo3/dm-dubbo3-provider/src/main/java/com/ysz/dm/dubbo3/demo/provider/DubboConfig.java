package com.ysz.dm.dubbo3.demo.provider;

import com.ysz.dm.dubbo3.demo.DemoService;
import org.apache.dubbo.config.ApplicationConfig;
import org.apache.dubbo.config.ProtocolConfig;
import org.apache.dubbo.config.ProviderConfig;
import org.apache.dubbo.config.RegistryConfig;
import org.apache.dubbo.config.spring.ServiceBean;
import org.apache.dubbo.config.spring.context.annotation.EnableDubbo;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
@EnableDubbo
public class DubboConfig {

  @Value("${application.name:demo-provider}")
  private String applicationName;


  @Bean
  public ApplicationConfig applicationConfig() {
    ApplicationConfig applicationConfig = new ApplicationConfig();
    applicationConfig.setName("demo-provider");
    return applicationConfig;
  }


  @Bean
  public RegistryConfig registryConfig() {
    RegistryConfig registryConfig = new RegistryConfig();
    registryConfig.setAddress("127.0.0.1:2181");
    registryConfig.setProtocol("zookeeper");
    registryConfig.setFile("/duitang/tmp/" + applicationName + "/dubbo");
    registryConfig.setRegister(false);
    return registryConfig;
  }

  @Bean
  public ProviderConfig providerConfig() {
    ProviderConfig providerConfig = new ProviderConfig();
    providerConfig.setToken(true);
    return providerConfig;

  }

  @Bean
  public DemoService demoService() {
    return new DemoServiceImpl();
  }
//
  @Bean
  public ProtocolConfig protocolConfig() {
      ProtocolConfig protocolConfig = new ProtocolConfig();
      protocolConfig.setName("dubbo");
      protocolConfig.setPort(10001);
      protocolConfig.setHost("127.0.0.1");
      return protocolConfig;
  }


  @Bean
  public ServiceBean<DemoService> demoServiceServiceBean(
      DemoService demoService
  ) {
    ServiceBean serviceBean = new ServiceBean();
    serviceBean.setSerialization("protobuf");
    serviceBean.setInterface("com.ysz.dm.dubbo3.demo.DemoService");
    serviceBean.setRef(demoService);
    return serviceBean;
  }

//  @Bean
//  public DubboBootstrapApplicationListener bootstrap() {
//    return new DubboBootstrapApplicationListener();
//  }
}
