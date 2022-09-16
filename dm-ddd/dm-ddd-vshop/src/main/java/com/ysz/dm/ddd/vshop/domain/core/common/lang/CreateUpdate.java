package com.ysz.dm.ddd.vshop.domain.core.common.lang;

import java.time.Instant;
import lombok.Getter;
import lombok.ToString;

/**
 * <pre>
 * class desc here
 * </pre>
 *
 * @author carl.yu
 * @createAt 2022/9/9
 **/
@ToString
@Getter
public class CreateUpdate {

  /**
   * 创建时间
   */
  private final Instant createAt;

  /**
   * 修改时间
   */
  private Instant updateAt;

  /**
   * 修改人
   */
  private String updateBy;


  public CreateUpdate(Instant createAt, Instant updateAt, String updateBy) {
    this.createAt = createAt;
    this.updateAt = updateAt;
    this.updateBy = updateBy;
  }

  public CreateUpdate setUpdateAt(Instant updateAt) {
    this.updateAt = updateAt;
    return this;
  }

  public CreateUpdate setUpdateBy(String updateBy) {
    this.updateBy = updateBy;
    return this;
  }
}
