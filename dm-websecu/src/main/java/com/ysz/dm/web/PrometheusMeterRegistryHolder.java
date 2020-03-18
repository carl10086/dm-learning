package com.ysz.dm.web;

import io.micrometer.core.instrument.binder.jvm.ClassLoaderMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmGcMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmMemoryMetrics;
import io.micrometer.core.instrument.binder.jvm.JvmThreadMetrics;
import io.micrometer.core.instrument.binder.logging.LogbackMetrics;
import io.micrometer.core.instrument.binder.system.FileDescriptorMetrics;
import io.micrometer.core.instrument.binder.system.ProcessorMetrics;
import io.micrometer.core.instrument.binder.system.UptimeMetrics;
import io.micrometer.prometheus.PrometheusConfig;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import io.micrometer.prometheus.PrometheusRenameFilter;

/**
 * @author carl.yu
 * @date 2020/3/18
 */
public class PrometheusMeterRegistryHolder {

  private PrometheusMeterRegistryHolder() {
  }

  public static PrometheusMeterRegistry getInstance() {
    return SingletonHolder.prometheusRegistry;
  }

  private static class SingletonHolder {

    static PrometheusMeterRegistry prometheusRegistry;

    static {
      prometheusRegistry = new PrometheusMeterRegistry(
          PrometheusConfig.DEFAULT);
      prometheusRegistry.config().commonTags("application", "dm-learning")
          .meterFilter(new PrometheusRenameFilter())
      ;
      new ClassLoaderMetrics().bindTo(prometheusRegistry);
      new JvmMemoryMetrics().bindTo(prometheusRegistry);
      new JvmGcMetrics().bindTo(prometheusRegistry);
      new ProcessorMetrics().bindTo(prometheusRegistry);
      new JvmThreadMetrics().bindTo(prometheusRegistry);
      new LogbackMetrics().bindTo(prometheusRegistry);
      new UptimeMetrics().bindTo(prometheusRegistry);
      new FileDescriptorMetrics().bindTo(prometheusRegistry);
    }
  }
}
