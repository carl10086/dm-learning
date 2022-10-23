package com.ysz.arm.blog.server.config;

import static java.util.Objects.requireNonNull;

import com.linecorp.armeria.server.Server;
import com.linecorp.armeria.spring.ArmeriaServerSmartLifecycle;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.context.SmartLifecycle;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/10/23
 **/
public final class CustomArmeriaServerGracefulShutdownLifecycle
    implements ArmeriaServerSmartLifecycle {

  private static final Logger logger = LoggerFactory.getLogger(CustomArmeriaServerGracefulShutdownLifecycle.class);

  private final Server server;
  private volatile boolean running;

  public CustomArmeriaServerGracefulShutdownLifecycle(Server server) {
    this.server = requireNonNull(server, "server");
  }

  /**
   * Start this component.
   */
  @Override
  public void start() {
    server.start().handle((result, t) -> {
      if (t != null) {
        throw new IllegalStateException("Armeria server failed to start", t);
      }
      running = true;
      return null;
    }).join();
    logger.info("Armeria server started at ports: {}", server.activePorts());
  }

  /**
   * Stop this component. This class implements {@link SmartLifecycle}, so don't need to support sync stop.
   */
  @Override
  public void stop() {
    throw new UnsupportedOperationException("Stop must not be invoked directly");
  }

  /**
   * Stop this component.
   */
  @Override
  public void stop(Runnable callback) {
    server.stop().whenComplete((unused, throwable) -> {
      callback.run();
      running = false;
    });
  }

  /**
   * Returns the phase that this lifecycle object is supposed to run in. WebServerStartStopLifecycle's phase is
   * Integer.MAX_VALUE - 1. To run before the tomcat, we need to larger than Integer.MAX_VALUE - 1.
   */
  @Override
  public int getPhase() {
    return Integer.MAX_VALUE;
  }

  /**
   * Check whether this component is currently running.
   */
  @Override
  public boolean isRunning() {
    return running;
  }

  /**
   * Returns true if this Lifecycle component should get started automatically by the container at the time that the
   * containing ApplicationContext gets refreshed.
   */
  @Override
  public boolean isAutoStartup() {
    return true;
  }

}
