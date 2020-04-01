package com.ysz.dm.agent;

import java.lang.instrument.Instrumentation;
import net.bytebuddy.agent.builder.AgentBuilder;
import net.bytebuddy.implementation.MethodDelegation;
import net.bytebuddy.matcher.ElementMatchers;

/**
 * @author carl.yu
 * @date 2020/3/19
 */
public class MyAgent {

  public static void premain(String agentArgs, Instrumentation inst) {

  }

}
