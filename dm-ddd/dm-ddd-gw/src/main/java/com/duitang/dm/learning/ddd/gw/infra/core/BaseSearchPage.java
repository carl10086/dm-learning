package com.duitang.dm.learning.ddd.gw.infra.core;

import java.io.Serializable;
import lombok.Getter;
import lombok.ToString;

@ToString
@Getter
public class BaseSearchPage implements Serializable {

  private static final long serialVersionUID = -6857021088408829474L;

  private String from;

  public BaseSearchPage setFrom(String from) {
    this.from = from;
    return this;
  }
}
