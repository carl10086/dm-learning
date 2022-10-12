package com.ysz.codemaker.patterns.builder.core;

import java.util.List;

/**
 * @author carl
 * @create 2022-10-12 4:35 PM
 **/
public record Context(
    /*simple class name*/
    String name,
    /*all fields*/
    List<Field> fields,
    List<String> imports
) {


}
