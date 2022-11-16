package com.ysz.dm.rb.user.infra.config

import com.ysz.dm.rb.user.infra.base.armeria.ArmeriaGrpc
import io.grpc.BindableService
import org.slf4j.LoggerFactory
import org.springframework.context.ApplicationContext
import org.springframework.context.ApplicationContextAware
import org.springframework.context.annotation.Configuration

/**
 * @author carl
 * @create 2022-11-16 5:22 PM
 **/
@Configuration
open class ArmeriaConfig : ApplicationContextAware {

    private var applicationContext: ApplicationContext? = null


    private fun grpcService(): com.linecorp.armeria.server.grpc.GrpcService {
        val grpcServiceBeanMap = this.applicationContext!!.getBeansWithAnnotation(ArmeriaGrpc::class.java)
        log.info("find ArmeriaGrpc beans size:{}", grpcServiceBeanMap.size)
        val builder = com.linecorp.armeria.server.grpc.GrpcService.builder()

        /*1. check and init grpc service*/for (value in grpcServiceBeanMap.values) {
            require(value is BindableService) { "service is not grpc service , class name:" + value.javaClass }
            builder.addService(value as io.grpc.BindableService)
        }

        /*2. support serialization formats*/builder.supportedSerializationFormats(com.linecorp.armeria.common.grpc.GrpcSerializationFormats.values())

        /*3. user blocking Task Executor*/builder.useBlockingTaskExecutor(true)
        return builder.build()
    }

    @org.springframework.context.annotation.Bean
    open fun configurator(): com.linecorp.armeria.spring.ArmeriaServerConfigurator? {
        return com.linecorp.armeria.spring.ArmeriaServerConfigurator { builder: com.linecorp.armeria.server.ServerBuilder ->
            // Add DocService that enables you to send Thrift and gRPC requests from web browser.
//            builder.serviceUnder("/docs", com.linecorp.armeria.server.docs.DocService())

            // Log every message which the server receives and responds.
//      builder.decorator(LoggingService.newDecorator());

            // Write access log after completing a request.
//      builder.accessLogWriter(AccessLogWriter.combined(), false);

            // Add an Armeria annotated HTTP service.
//      builder.annotatedService(service);
//            val zkConnectionStr: kotlin.String = "wvr-zk-67-4.duitang.net:3881"
//            val znodePath: kotlin.String = "/armeria/server"
//            val serviceName: kotlin.String = "catalog"
//            val registrationSpec: ZooKeeperRegistrationSpec = ZooKeeperRegistrationSpec.curator(serviceName)
//            val listener: ZooKeeperUpdatingListener =
//                ZooKeeperUpdatingListener.builder(zkConnectionStr, znodePath, registrationSpec)
//                    .sessionTimeoutMillis(10000)
//                    .build()

//      builder.serverListener(listener);
            builder.service(grpcService()).decorator(
                com.linecorp.armeria.server.metric.MetricCollectingService.newDecorator(
                    com.linecorp.armeria.common.grpc.GrpcMeterIdPrefixFunction.of("grpc.service")
                )
            )
        }
    }


    companion object {
        private val log = LoggerFactory.getLogger(ArmeriaConfig::class.java)
    }

    override fun setApplicationContext(applicationContext: ApplicationContext) {
        this.applicationContext = applicationContext
    }
}