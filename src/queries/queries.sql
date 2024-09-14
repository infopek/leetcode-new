-- https://leetcode.com/problems/combine-two-tables/
select firstName, lastName, city, state
from Person p left join Address a
on p.personId = a.personId;

-- https://leetcode.com/problems/employees-earning-more-than-their-managers/
select e1.name as Employee
from Employee e1 inner join Employee e2
on e1.managerId = e2.id
where e1.salary > e2.salary;

-- https://leetcode.com/problems/duplicate-emails/
select email as Email
from Person
group by email
having count(email) > 1;

-- https://leetcode.com/problems/customers-who-never-order/
select c.name as Customers
from Customers c left join Orders o
on c.id = o.customerId
where o.customerId is null;

-- https://leetcode.com/problems/delete-duplicate-emails/
delete p1 from Person p1, Person p2
where p1.email = p2.email and p1.id > p2.id;

-- https://leetcode.com/problems/rising-temperature/
select w1.id
from Weather w1, Weather w2
where datediff(w1.recordDate, w2.recordDate) = 1
and w1.temperature > w2.temperature;

-- https://leetcode.com/problems/game-play-analysis-i/
-- TTS: 05:01
select a1.player_id, min(a1.event_date) as first_login
from Activity a1
group by a1.player_id;

-- https://leetcode.com/problems/employee-bonus/
-- TTS: 03:07
select name, bonus
from Employee e left join Bonus b on e.empId = b.empId
where bonus < 1000 or b.empId is null;

-- https://leetcode.com/problems/find-customer-referee/
-- TTS: 01:16
select name
from Customer
where referee_id is null or referee_id != 2;

-- https://leetcode.com/problems/customer-placing-the-largest-number-of-orders/
-- TTS: 31:42
{
    -- Solution 1
    select customer_number
    from Orders
    group by customer_number
    order by count(customer_number) desc
    limit 1;

    -- Solution 2: follow-up using subqueries
    select customer_number
    from Orders
    group by customer_number
    having count(order_number) = 
    (
        select max(o_n)
        from 
        (
            select count(order_number) as o_n
            from Orders
            group by customer_number
        ) as sub
    );

    -- Solution 2: follow-up using cte
    with cte as
    (
        select customer_number,
        rank() over(order by count(order_number) desc) as order_rank
        from orders
        group by customer_number
    )

    select customer_number from cte
    where order_rank = 1;
}

-- https://leetcode.com/problems/big-countries/
-- TTS: 11:15
{
    -- Solution 1: simple
    select name, population, area
    from World
    where area >= 3000000/ or population >= 25000000;

    -- Solution 2: union
    select name, population, area
    from World
    where area >= 3000000
    union
    select name, population, area
    from World
    where population >= 25000000;
}

-- https://leetcode.com/problems/classes-more-than-5-students/
-- TTS: 03:12
select class
from Courses
group by class
having count(class) >= 5;

-- https://leetcode.com/problems/sales-person/
-- TTS: 27:14
select sp.name
from Orders o
right join Company c on (o.com_id = c.com_id and c.name = 'RED')
right join SalesPerson sp on sp.sales_id = o.sales_id
where o.sales_id is null;

-- https://leetcode.com/problems/triangle-judgement/
-- TTS: 06:03
select *, 
case
    when x + y > z and x + z > y and y + z > x then 'Yes'
    else 'No'
end as triangle
from Triangle;

-- https://leetcode.com/problems/biggest-single-number/
-- TTS: 02:06
{
    -- Solution 1: fast
    select max(sub.num) as num
    from
    (
        select num
        from MyNumbers n
        group by n.num
        having count(n.num) = 1
    ) as sub;

    -- Solution 2: without subquery
    select if (count(*) = 1, num, null) as num
    from MyNumbers
    group by num
    order by count(*), num desc
    limit 1;
}

-- https://leetcode.com/problems/not-boring-movies/
-- TTS: 01:03
select *
from Cinema
where description != "boring"
and mod(id, 2) = 1
order by rating desc;

-- https://leetcode.com/problems/swap-salary/
-- TTS: 03:42
{
    -- Solution 1: case
    update Salary
    set sex = case sex
        when 'm' then 'f'
        else 'm'
    end;

    -- Solution 2: ternary operator
    update Salary
    set sex = if(sex = 'm', 'f', 'm');

    -- Solution 3: replace()
    update Salary
    set sex = replace('fm', sex, '');

    -- Solution 4: math
    update Salary
    set sex = char(ascii('f') + ascii('m') - ascii(sex));

    -- Solution 5: xor
    update Salary
    set sex = char(ascii('f') ^ ascii('m') ^ ascii(sex));
}