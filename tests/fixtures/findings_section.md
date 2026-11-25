## Confirmed bugs (2)

1. **Total ignores discount: Cart total shows the undiscounted price**
   - Observed: $120.00
   - Expected: $96.00
   - Where: [data-test="total"] on /cart
   - Severity: major
2. **Empty cart lacks messaging: Cart page shows no guidance text**
   - Observed: no observed value recorded
   - Where: /cart
   - Severity: minor

## Not determined (1)

The agent could not confirm these. They are not counted as bugs.

1. **Coupon may reject lowercase codes: SAVE20 applied but save20 seemed to do nothing**
   - Observed: no change after save20
   - Severity: minor
