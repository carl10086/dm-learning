package com.ysz.arm.blog.server.config;

import static com.linecorp.armeria.internal.spring.ArmeriaConfigurationUtil.configureServerWithArmeriaSettings;

import com.google.common.collect.ImmutableList;
import com.linecorp.armeria.common.DependencyInjector;
import com.linecorp.armeria.common.SessionProtocol;
import com.linecorp.armeria.common.annotation.Nullable;
import com.linecorp.armeria.common.metric.MeterIdPrefixFunction;
import com.linecorp.armeria.server.Server;
import com.linecorp.armeria.server.ServerBuilder;
import com.linecorp.armeria.server.ServerPort;
import com.linecorp.armeria.server.healthcheck.HealthChecker;
import com.linecorp.armeria.spring.ArmeriaServerConfigurator;
import com.linecorp.armeria.spring.ArmeriaServerSmartLifecycle;
import com.linecorp.armeria.spring.ArmeriaSettings;
import com.linecorp.armeria.spring.ArmeriaSettings.Port;
import com.linecorp.armeria.spring.DocServiceConfigurator;
import com.linecorp.armeria.spring.HealthCheckServiceConfigurator;
import com.linecorp.armeria.spring.InternalServices;
import com.linecorp.armeria.spring.MetricCollectingServiceConfigurator;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Metrics;
import java.time.Duration;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import org.springframework.beans.factory.BeanFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.SmartLifecycle;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * <pre>
 * custom armeria configuration
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/23
 **/
//@Configuration
//@ConditionalOnProperty(name = "armeria.server-enabled", havingValue = "true", matchIfMissing = true)
//@EnableConfigurationProperties(ArmeriaSettings.class)
public class CustomArmeriaAutoConfiguration {

  private static final Port DEFAULT_PORT = new Port().setPort(8080)
      .setProtocol(SessionProtocol.HTTP);

  private static final String GRACEFUL_SHUTDOWN = "graceful";

  @Bean
  public InternalServices internalServices(
      ArmeriaSettings settings,
      Optional<MeterRegistry> meterRegistry,
      Optional<List<HealthChecker>> healthCheckers,
      Optional<List<HealthCheckServiceConfigurator>> healthCheckServiceConfigurators,
      Optional<List<DocServiceConfigurator>> docServiceConfigurators,
      @Value("${management.server.port:#{null}}") @Nullable Integer managementServerPort
  ) {

    return InternalServices.of(settings, meterRegistry.orElse(Metrics.globalRegistry),
                               healthCheckers.orElse(ImmutableList.of()),
                               healthCheckServiceConfigurators.orElse(ImmutableList.of()),
                               docServiceConfigurators.orElse(ImmutableList.of()), managementServerPort
    );
  }


  @Bean
  @ConditionalOnMissingBean(Server.class)
  public Server armeriaServer(
      ArmeriaSettings armeriaSettings,
      InternalServices internalService,
      Optional<MeterRegistry> meterRegistry,
      Optional<List<MetricCollectingServiceConfigurator>> metricCollectingServiceConfigurators,
      Optional<MeterIdPrefixFunction> meterIdPrefixFunction,
      Optional<List<ArmeriaServerConfigurator>> armeriaServerConfigurators /*这里是暴露出去的配置*/,
      Optional<List<Consumer<ServerBuilder>>> armeriaServerBuilderConsumers,
      Optional<List<DependencyInjector>> dependencyInjectors,
      BeanFactory beanFactory
  ) {

    if (!armeriaServerConfigurators.isPresent() &&
        !armeriaServerBuilderConsumers.isPresent()) {
      throw new IllegalStateException(
          "No services to register, " +
              "use ArmeriaServerConfigurator or Consumer<ServerBuilder> to configure an Armeria server.");
    }

    final ServerBuilder serverBuilder = Server.builder();

    final List<Port> ports = armeriaSettings.getPorts(); /*init default ports*/
    if (ports.isEmpty()) {
      assert DEFAULT_PORT.getProtocols() != null;
      serverBuilder.port(new ServerPort(DEFAULT_PORT.getPort(), DEFAULT_PORT.getProtocols()));
    }

    configureServerWithArmeriaSettings(serverBuilder,
                                       armeriaSettings,
                                       internalService,
                                       armeriaServerConfigurators.orElse(ImmutableList.of()),
                                       armeriaServerBuilderConsumers.orElse(ImmutableList.of()),
                                       meterRegistry.orElse(Metrics.globalRegistry),
                                       meterIdPrefixFunction.orElse(
                                           MeterIdPrefixFunction.ofDefault("armeria.server")),
                                       metricCollectingServiceConfigurators.orElse(ImmutableList.of()),
                                       dependencyInjectors.orElse(ImmutableList.of()),
                                       beanFactory
    );

    return serverBuilder.build();
  }

  @Bean
  @ConditionalOnMissingBean(ArmeriaServerSmartLifecycle.class)
  public SmartLifecycle armeriaServerGracefulShutdownLifecycle(Server server) {
    return new CustomArmeriaServerGracefulShutdownLifecycle(server);
  }

  @Bean
  @ConditionalOnProperty("server.shutdown")
  public ArmeriaServerConfigurator gracefulShutdownServerConfigurator(
      @Value("${server.shutdown}") String shutdown,
      @Value("${spring.lifecycle.timeout-per-shutdown-phase:30s}") Duration duration
  ) {
    if (GRACEFUL_SHUTDOWN.equalsIgnoreCase(shutdown)) {
      return sb -> sb.gracefulShutdownTimeout(duration, duration);
    } else {
      return sb -> {
      };
    }
  }
}
