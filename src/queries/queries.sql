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