package com.ysz.dm.netty.dm.echo;

import java.io.Serializable;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class SimpleStringWrapper implements Serializable {

  private static final long serialVersionUID = 1L;

  private String val;

}
