package com.ysz.dm.netty.order.domain;

import com.ysz.dm.netty.order.domain.order.OrderOperation;
import com.ysz.dm.netty.order.domain.order.OrderOperationResult;
import java.util.Objects;
import java.util.function.Predicate;
import java.util.stream.Stream;
import lombok.Getter;

/**
 * @author carl
 */

@Getter
public enum OperationType {
  /**
   * Tst Order Operation
   */
  ORDER(3, OrderOperation.class, OrderOperationResult.class);

  private final Integer opCode;
  private final Class<? extends Operation> operationClazz;
  private final Class<? extends OperationResult> operationResultClazz;


  OperationType(Integer opCode,
      Class<? extends Operation> operationClazz,
      Class<? extends OperationResult> operationResultClazz) {
    this.opCode = opCode;
    this.operationClazz = operationClazz;
    this.operationResultClazz = operationResultClazz;
  }


  public static OperationType fromOperation(Operation operation) {
    return of(x -> x.getOperationClazz() == operation.getClass());
  }

  public static OperationType fromOpCode(final Integer code) {
    return of(x -> Objects.equals(code, x.getOpCode()));
  }

  private static OperationType of(Predicate<OperationType> predicate) {
    return Stream.of(OperationType.values()).filter(predicate).findAny().orElse(null);
  }

}
