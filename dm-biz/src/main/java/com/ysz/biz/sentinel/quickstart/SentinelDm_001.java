package com.ysz.biz.sentinel.quickstart;

import com.alibaba.csp.sentinel.Entry;
import com.alibaba.csp.sentinel.SphU;
import com.alibaba.csp.sentinel.slots.block.BlockException;
import com.alibaba.csp.sentinel.slots.block.RuleConstant;
import com.alibaba.csp.sentinel.slots.block.flow.FlowRule;
import com.alibaba.csp.sentinel.slots.block.flow.FlowRuleManager;
import java.util.ArrayList;
import java.util.List;

public class SentinelDm_001 {

  private static void initFlowRules() {
    List<FlowRule> rules = new ArrayList<>();

    /*创建了一条规则, 核心就是这个资源的限制 qps 是20*/
    FlowRule rule = new FlowRule();

    /*资源名称*/
    rule.setResource("HelloWorld");
    rule.setGrade(RuleConstant.FLOW_GRADE_QPS);
    // Set limit QPS to 20.
    rule.setCount(20);
    rules.add(rule);

    FlowRuleManager.loadRules(rules);
  }

  public static void main(String[] args) {
    initFlowRules();

    while (true) {
      Entry entry = null;
      try {
        /*根据名称找规则、发现是 qps 20*/
        entry = SphU.entry("HelloWorld");
        /*您的业务逻辑 - 开始*/
        System.out.println("hello world");
        /*您的业务逻辑 - 结束*/
      } catch (BlockException e1) {
        /*流控逻辑处理 - 开始*/
        System.out.println("block!");
        /*流控逻辑处理 - 结束*/
      } finally {
        if (entry != null) {
          entry.exit();
        }
      }
    }

  }

}
