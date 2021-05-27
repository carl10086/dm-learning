package com.ysz.biz.easy.rule;

import org.jeasy.rules.api.Facts;
import org.jeasy.rules.api.Rules;
import org.jeasy.rules.api.RulesEngine;
import org.jeasy.rules.core.DefaultRulesEngine;
import org.jeasy.rules.core.RuleBuilder;

public class BasicUsage {

  public static void main(String[] args) throws Exception {
    Rules rules = new Rules();
    rules.register(new RuleBuilder()
        .name("weather rule")
        .description("if it rains then take an umbrella")
        .when(facts1 -> facts1.get("rain").equals(false))
        .then(facts1 -> System.out.println("It rains, take an umbrella!"))
        .build());

    /*事实*/
    Facts facts = new Facts();
    facts.put("rain", false);

    RulesEngine rulesEngine = new DefaultRulesEngine();
    rulesEngine.fire(rules, facts);
  }


}
