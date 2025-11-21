/*
 * Copyright 2022 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */
.sram2_noncachable (NOLOAD):
{
  __sram2_noinit_start = .;
  . = ALIGN(32);
  *(.NonCacheable.ZeroInit)
  *(.bss.vram.data)
  . = ALIGN(32);
  __sram2_noinit_end = .;
} > SRAM2 AT > SRAM2
