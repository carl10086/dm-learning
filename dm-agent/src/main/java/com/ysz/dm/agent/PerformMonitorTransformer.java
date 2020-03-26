package com.ysz.dm.agent;

import java.lang.instrument.ClassFileTransformer;
import java.lang.instrument.IllegalClassFormatException;
import java.security.ProtectionDomain;
import java.util.HashSet;
import java.util.Set;

/**
 * @author carl.yu
 * @date 2020/3/19
 */
public class PerformMonitorTransformer implements ClassFileTransformer {

  private static final Set<String> classNameSet = new HashSet<>();

  static {
    classNameSet.add("com.example.demo.AgentTest");
  }

  @Override
  public byte[] transform(ClassLoader loader, String className, Class<?> classBeingRedefined,
      ProtectionDomain protectionDomain, byte[] classfileBuffer)
      throws IllegalClassFormatException {
    return new byte[0];
  }
}
