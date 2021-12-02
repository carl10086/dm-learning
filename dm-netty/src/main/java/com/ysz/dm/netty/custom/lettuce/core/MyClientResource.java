package com.ysz.dm.netty.custom.lettuce.core;

import com.google.common.base.Preconditions;
import com.ysz.dm.netty.custom.netty.core.channel.eventloop.MyEventExecutorGroup;
import io.lettuce.core.resource.Delay;
import io.netty.util.Timer;
import io.netty.util.concurrent.EventExecutorGroup;
import io.netty.util.internal.SystemPropertyUtil;
import io.netty.util.internal.logging.InternalLogger;
import io.netty.util.internal.logging.InternalLoggerFactory;
import java.util.function.Supplier;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class MyClientResource {

  private static final InternalLogger logger = InternalLoggerFactory
      .getInstance(MyClientResource.class);


  /**
   * Minimum number of I/O threads.
   */
  public static final int MIN_IO_THREADS = 2;

  /**
   * Minimum number of computation threads.
   */
  public static final int MIN_COMPUTATION_THREADS = 2;

  public static final int DEFAULT_IO_THREADS;

  public static final int DEFAULT_COMPUTATION_THREADS;

  static {
    int threads = Math.max(1, SystemPropertyUtil.getInt("io.netty.eventLoopThreads",
        Math.max(MIN_IO_THREADS, Runtime.getRuntime().availableProcessors())));

    /**
     * 计算默认的 IO THREAD 数目
     */
    DEFAULT_IO_THREADS = threads;
    DEFAULT_COMPUTATION_THREADS = threads;

    if (logger.isDebugEnabled()) {
      logger.debug("-Dio.netty.eventLoopThreads: {}", threads);
    }
  }

  private MyCommandLatencyRecorder commandLatencyRecorder;

  private boolean sharedCommandLatencyRecorder;

  private MyEventPublisherOptions commandLatencyPublisherOptions;

  private MyDnsResolver dnsResolver;

  private MyEventBus eventBus;

  private boolean sharedEventLoopGroupProvider;

  private MyEventLoopGroupProvider eventLoopGroupProvider;

  private boolean sharedEventExecutor;

  private EventExecutorGroup eventExecutorGroup;

  private MetricEventPublisher metricEventPublisher;

  private MyNettyCustomizer nettyCustomizer;

  private Supplier<Delay> reconnectDelay;

  private MySocketAddressResolver socketAddressResolver;

  private Timer timer;

  private boolean sharedTimer;

  private MyTracing tracing;

  private boolean shutdownCalled;

  public static class Builder {

    private MyCommandLatencyRecorder commandLatencyRecorder;

    private boolean sharedCommandLatencyRecorder;

    private MyEventPublisherOptions commandLatencyPublisherOptions;

    private MyDnsResolver dnsResolver;

    private MyEventBus eventBus;

    private boolean sharedEventLoopGroupProvider;

    private int ioThreadPoolSize = DEFAULT_IO_THREADS;

    private MyEventLoopGroupProvider eventLoopGroupProvider;

    private boolean sharedEventExecutor;

    private MyEventExecutorGroup eventExecutorGroup;

    private MetricEventPublisher metricEventPublisher;

    private MyNettyCustomizer nettyCustomizer;

    private Supplier<Delay> reconnectDelay;

    private MySocketAddressResolver socketAddressResolver;

    private Timer timer;

    private boolean sharedTimer;

    private MyTracing tracing;

    private boolean shutdownCalled;

    private boolean sharedCommandLatencyCollector;


    private int computationThreadPoolSize = DEFAULT_COMPUTATION_THREADS;

    private Builder() {
    }


    public Builder commandLatencyPublisherOptions(
        MyEventPublisherOptions commandLatencyPublisherOptions
    ) {
      this.commandLatencyPublisherOptions = Preconditions.checkNotNull(
          commandLatencyPublisherOptions, "EventPublisherOptions must not be null"
      );

      return this;
    }


    public Builder commandLatencyRecorder(MyCommandLatencyRecorder commandLatencyRecorder) {
      this.sharedCommandLatencyCollector = true;
      this.commandLatencyRecorder = Preconditions.checkNotNull(
          commandLatencyRecorder,
          "CommandLatencyRecorder must not be null"
      );

      return this;
    }


    public Builder computationThreadPoolSize(
        int computationThreadPoolSize
    ) {
      Preconditions.checkState(
          computationThreadPoolSize > 0,
          "Computation thread pool size must be greater zero"
      );

      this.computationThreadPoolSize = computationThreadPoolSize;
      return this;
    }

    public Builder dnsResolver(
        MyDnsResolver dnsResolver
    ) {
      this.dnsResolver = Preconditions.checkNotNull(
          dnsResolver,
          "DnsResolver must not be null"
      );

      return this;
    }


    public Builder eventBus(MyEventBus eventBus) {
      this.eventBus = Preconditions.checkNotNull(eventBus, "EventBus must not be null");
      return this;
    }


    public Builder eventLoopGroupProvider(
        MyEventLoopGroupProvider eventLoopGroupProvider
    ) {
      this.eventLoopGroupProvider = Preconditions.checkNotNull(
          eventLoopGroupProvider,
          "eventLoopGroupProvider can't be null"
      );
      this.sharedEventLoopGroupProvider = true;
      return this;
    }

    public Builder eventExecutorGroup(
        MyEventExecutorGroup eventExecutorGroup
    ) {
      this.sharedEventExecutor = true;
      this.eventExecutorGroup = Preconditions.checkNotNull(
          eventExecutorGroup, "EventExecutorGroup must not be null"
      );

      return this;
    }

    public Builder nettyCustomizer(MyNettyCustomizer nettyCustomizer) {
      this.nettyCustomizer = Preconditions.checkNotNull(
          nettyCustomizer, "NettyCustomizer must not be null"
      );
      return this;
    }

    public Builder ioThreadPoolSize(int ioThreadPoolSize) {
      Preconditions.checkState(ioThreadPoolSize > 0, "I/O thread pool size must be greater zero");
      this.ioThreadPoolSize = ioThreadPoolSize;
      return this;
    }


    public Builder timer(Timer timer) {
      this.timer = Preconditions.checkNotNull(
          timer, "Timer must not be null"
      );

      this.sharedTimer = true;
      return this;
    }

    public Builder tracing(MyTracing tracing) {
      this.tracing = Preconditions.checkNotNull(
          tracing, "Tracing must not be null"
      );
      return this;
    }
  }
}
