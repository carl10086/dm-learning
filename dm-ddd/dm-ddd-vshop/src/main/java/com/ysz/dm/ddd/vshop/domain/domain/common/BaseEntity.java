package com.ysz.dm.ddd.vshop.domain.domain.common;

import java.time.Instant;

/**
 * @author carl
 * @create 2022-10-24 5:31 PM
 **/
public abstract class BaseEntity {

  private Instant updateAt;
  private Instant createAt;
  private String updateBy;
  private String createBy;

  public BaseEntity() {
  }

  public Instant getUpdateAt() {
    return updateAt;
  }

  public BaseEntity setUpdateAt(Instant updateAt) {
    this.updateAt = updateAt;
    return this;
  }

  public Instant getCreateAt() {
    return createAt;
  }

  public BaseEntity setCreateAt(Instant createAt) {
    this.createAt = createAt;
    return this;
  }

  public String getUpdateBy() {
    return updateBy;
  }

  public BaseEntity setUpdateBy(String updateBy) {
    this.updateBy = updateBy;
    return this;
  }

  public BaseEntity setCreateBy(String createBy) {
    this.createBy = createBy;
    return this;
  }
}
