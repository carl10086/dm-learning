package com.ysz.dm.netty.dm.order.domain;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * @author carl
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class MessageHeader {

  private Integer version = -1;

  private Integer opCode;

  private Long streamId;

}
