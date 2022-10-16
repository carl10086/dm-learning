package com.ysz.arm.blog.server.config;

import com.linecorp.armeria.common.grpc.GrpcSerializationFormats;
import com.linecorp.armeria.server.docs.DocService;
import com.linecorp.armeria.server.grpc.GrpcService;
import com.linecorp.armeria.server.grpc.GrpcServiceBuilder;
import com.linecorp.armeria.server.zookeeper.ZooKeeperRegistrationSpec;
import com.linecorp.armeria.server.zookeeper.ZooKeeperUpdatingListener;
import com.linecorp.armeria.server.zookeeper.ZooKeeperUpdatingListenerBuilder;
import com.linecorp.armeria.spring.ArmeriaServerConfigurator;
import io.grpc.BindableService;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.BeansException;
import org.springframework.context.ApplicationContext;
import org.springframework.context.ApplicationContextAware;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/16
 **/
@Configuration
@Slf4j
public class ArmeriaConfig implements ApplicationContextAware {

  private ApplicationContext applicationContext;

  private GrpcService grpcService() {
    Map<String, Object> grpcServiceBeanMap = this.applicationContext.getBeansWithAnnotation(ArmeriaGrpc.class);
    log.info("find ArmeriaGrpc beans size:{}", grpcServiceBeanMap.size());

    GrpcServiceBuilder builder = GrpcService.builder();

    /*1. check and init grpc service*/
    for (Object value : grpcServiceBeanMap.values()) {
      if (!(value instanceof BindableService)) {
        throw new IllegalArgumentException("service is not grpc service , class name:" + value.getClass());
      }
      builder.addService((BindableService) value);
    }

    /*2. support serialization formats*/
    builder.supportedSerializationFormats(GrpcSerializationFormats.values());

    /*3. user blocking Task Executor*/
    builder.useBlockingTaskExecutor(true);

    return builder.build();
  }

  @Bean
  public GrpcServiceAspect monitorAspect() {
    return new GrpcServiceAspect();
  }


  @Bean
  public ArmeriaServerConfigurator configurator() {

    return builder -> {
      // Add DocService that enables you to send Thrift and gRPC requests from web browser.
      builder.serviceUnder("/docs", new DocService());

      // Log every message which the server receives and responds.
//      builder.decorator(LoggingService.newDecorator());

      // Write access log after completing a request.
//      builder.accessLogWriter(AccessLogWriter.combined(), false);

      // Add an Armeria annotated HTTP service.
//      builder.annotatedService(service);
      builder.blockingTaskExecutor(300);
      builder.verboseResponses(true);

      String zkConnectionStr = "wvr-zk-67-4.duitang.net:3881";
      String znodePath = "/armeria/server";
      String serviceName = "catalog";
      ZooKeeperRegistrationSpec registrationSpec =
          ZooKeeperRegistrationSpec.curator(serviceName);

      ZooKeeperUpdatingListener listener =
          ZooKeeperUpdatingListener.builder(zkConnectionStr, znodePath, registrationSpec)
              .sessionTimeoutMillis(10000)
              .build();

      builder.serverListener(listener);
      builder.service(grpcService());
    };
  }

  @Override
  public void setApplicationContext(ApplicationContext applicationContext) throws BeansException {
    this.applicationContext = applicationContext;


  }
}
