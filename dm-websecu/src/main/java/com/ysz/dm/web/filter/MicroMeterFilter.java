package com.ysz.dm.web.filter;

import com.ysz.dm.web.PrometheusMeterRegistryHolder;
import io.micrometer.core.instrument.Timer;
import io.micrometer.prometheus.PrometheusMeterRegistry;
import java.io.IOException;
import javax.servlet.Filter;
import javax.servlet.FilterChain;
import javax.servlet.FilterConfig;
import javax.servlet.ServletException;
import javax.servlet.ServletRequest;
import javax.servlet.ServletResponse;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpServletResponse;
import org.springframework.stereotype.Component;

/**
 * @author carl.yu
 * @date 2020/3/18
 */
@Component
public class MicroMeterFilter implements Filter {

  @Override
  public void init(FilterConfig filterConfig) throws ServletException {

  }

  @Override
  public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse,
      FilterChain filterChain) throws IOException, ServletException {
    PrometheusMeterRegistry prometheusRegistry = PrometheusMeterRegistryHolder.getInstance();
    Timer.Sample sample = Timer.start(prometheusRegistry);
    HttpServletRequest httpServletRequest = (HttpServletRequest) servletRequest;
    HttpServletResponse httpServletResponse = (HttpServletResponse) servletResponse;
    String requestURI = httpServletRequest.getRequestURI();
    try {
      filterChain.doFilter(servletRequest, servletResponse);
    } catch (Exception e) {
      throw e;
    } finally {
      sample.stop(
          Timer.builder("url_timer")
              .tags()
//              .publishPercentiles(0.95)
//              .publishPercentileHistogram()
//              .sla(Duration.ofMillis(100))
//              .minimumExpectedValue(Duration.ofMillis(50))
//              .maximumExpectedValue(Duration.ofMillis(2000))
              .tag("uri", requestURI)
              .tag("method", httpServletRequest.getMethod())
              .tag("status", httpServletResponse.getStatus() + "")
              .register(prometheusRegistry)
      );
    }
  }

  @Override
  public void destroy() {

  }
}
