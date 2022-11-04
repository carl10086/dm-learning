package com.ysz.dm.ddd.vshop.domain.domain.common;

import java.time.Instant;
import lombok.Getter;
import lombok.NonNull;

/**
 * @author carl
 * @create 2022-10-24 5:31 PM
 **/
@Getter
public abstract class BaseEntity<ID> {

  @NonNull
  private Instant updateAt;

  @NonNull
  private Instant createAt;

  private String updateBy;
  private String createBy;

  public BaseEntity() {
  }

  protected abstract ID id();


}
